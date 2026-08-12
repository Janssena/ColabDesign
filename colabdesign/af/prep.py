import jax
import jax.numpy as jnp
import numpy as np
import re

from colabdesign.af.alphafold.data import pipeline, prep_inputs, parsers
from colabdesign.af.alphafold.common import protein, residue_constants
from colabdesign.af.alphafold.model.tf import shape_placeholders
from colabdesign.af.alphafold.model import config


from colabdesign.shared.protein import _np_get_cb, pdb_to_string
from colabdesign.shared.prep import prep_pos
from colabdesign.shared.utils import copy_dict
from colabdesign.shared.model import order_aa

resname_to_idx = residue_constants.resname_to_idx
idx_to_resname = dict((v,k) for k,v in resname_to_idx.items())

#################################################
# AF_PREP - input prep functions
#################################################
class _af_prep:

  def _prep_model(self, **kwargs):
    '''prep model'''
    if not hasattr(self,"_model") or self._cfg != self._model["runner"].config:
      self._cfg.model.global_config.subbatch_size = None
      self._model = self._get_model(self._cfg)
      if sum(self._lengths) > 384:
        self._cfg.model.global_config.subbatch_size = 4
        self._model["fn"] = self._get_model(self._cfg)["fn"]

    self._opt = copy_dict(self.opt)  
    self.restart(**kwargs)

  def _prep_features(self, num_res, num_seq=None, num_templates=1):
    '''process features'''
    if num_seq is None: num_seq = self._num
    return prep_input_features(L=num_res, N=num_seq, T=num_templates)

  def _prep_fixbb(self, pdb_filename, chain=None,
                  copies=1, repeat=False, homooligomer=False,
                  rm_template=False,
                  rm_template_seq=True,
                  rm_template_sc=True,
                  rm_template_ic=False,
                  fix_pos=None, ignore_missing=True, **kwargs):
    '''
    prep inputs for fixed backbone design
    ---------------------------------------------------
    if copies > 1:
      -homooligomer=True - input pdb chains are parsed as homo-oligomeric units
      -repeat=True       - tie the repeating sequence within single chain
    -rm_template_seq     - if template is defined, remove information about template sequence
    -fix_pos="1,2-10"    - specify which positions to keep fixed in the sequence
                           note: supervised loss is applied to all positions, use "partial" 
                           protocol to apply supervised loss to only subset of positions
    -ignore_missing=True - skip positions that have missing density (no CA coordinate)
    ---------------------------------------------------
    '''    
    # prep features
    self._pdb = prep_pdb(pdb_filename, chain=chain, ignore_missing=ignore_missing,
                         offsets=kwargs.pop("pdb_offsets",None),
                         lengths=kwargs.pop("pdb_lengths",None))

    self._len = self._pdb["residue_index"].shape[0]
    self._lengths = [self._len]

    # feat dims
    num_seq = self._num
    res_idx = self._pdb["residue_index"]
    
    # get [pos]itions of interests    
    if fix_pos is not None and fix_pos != "":
      self._pos_info = prep_pos(fix_pos, **self._pdb["idx"])
      self.opt["fix_pos"] = self._pos_info["pos"]

    if homooligomer and chain is not None and copies == 1:
      copies = len(chain.split(","))
      
    # repeat/homo-oligomeric support
    if copies > 1:

      if repeat or homooligomer:
        self._len = self._len // copies
        if "fix_pos" in self.opt:
          self.opt["fix_pos"] = self.opt["fix_pos"][self.opt["fix_pos"] < self._len]

      if repeat:
        self._lengths = [self._len * copies]
        block_diag = False

      else:
        self._lengths = [self._len] * copies
        block_diag = not self._args["use_multimer"]

        res_idx = repeat_idx(res_idx[:self._len], copies)
        num_seq = (self._num * copies + 1) if block_diag else self._num
        self.opt["weights"].update({"i_pae":0.0, "i_con":0.0})

      self._args.update({"copies":copies, "repeat":repeat, "homooligomer":homooligomer, "block_diag":block_diag})
      homooligomer = not repeat
    else:
      self._lengths = self._pdb["lengths"]

    # configure input features
    self._inputs = self._prep_features(num_res=sum(self._lengths), num_seq=num_seq)
    self._inputs["residue_index"] = res_idx
    self._inputs["batch"] = make_fixed_size(self._pdb["batch"], num_res=sum(self._lengths))
    self._inputs.update(get_multi_id(self._lengths, homooligomer=homooligomer))

    # configure options/weights
    self.opt["weights"].update({"dgram_cce":1.0, "rmsd":0.0, "fape":0.0, "con":0.0})
    self._wt_aatype = self._inputs["batch"]["aatype"][:self._len]

    # configure template [opt]ions
    rm,L = {},sum(self._lengths)
    for n,x in {"rm_template":    rm_template,
                "rm_template_seq":rm_template_seq,
                "rm_template_sc": rm_template_sc}.items():
      rm[n] = np.full(L,False)
      if isinstance(x,str):
        rm[n][prep_pos(x,**self._pdb["idx"])["pos"]] = True
      else:
        rm[n][:] = x
    self.opt["template"]["rm_ic"] = rm_template_ic
    self._inputs.update(rm)
  
    self._prep_model(**kwargs)
    
  def _prep_hallucination(self, length=100, copies=1, repeat=False, **kwargs):
    '''
    prep inputs for hallucination
    ---------------------------------------------------
    if copies > 1:
      -repeat=True - tie the repeating sequence within single chain
    ---------------------------------------------------
    '''
    
    # define num copies (for repeats/ homo-oligomers)
    if not repeat and copies > 1 and not self._args["use_multimer"]:
      (num_seq, block_diag) = (self._num * copies + 1, True)
    else:
      (num_seq, block_diag) = (self._num, False)
    
    self._args.update({"repeat":repeat,"block_diag":block_diag,"copies":copies})
      
    # prep features
    self._len = length
    
    # set weights
    self.opt["weights"].update({"con":1.0})
    if copies > 1:
      if repeat:
        offset = 1
        self._lengths = [self._len * copies]
        self._args["repeat"] = True
      else:
        offset = 50
        self._lengths = [self._len] * copies
        self.opt["weights"].update({"i_pae":0.0, "i_con":1.0})
        self._args["homooligomer"] = True
      res_idx = repeat_idx(np.arange(length), copies, offset=offset)
    else:
      self._lengths = [self._len]
      res_idx = np.arange(length)
    
    # configure input features
    self._inputs = self._prep_features(num_res=sum(self._lengths), num_seq=num_seq)
    self._inputs["residue_index"] = res_idx
    self._inputs.update(get_multi_id(self._lengths, homooligomer=True))

    self._prep_model(**kwargs)

  def _prep_binder(self, pdb_filename,
                   target_chain="A", binder_len=50,                                         
                   rm_target = False,
                   rm_target_seq = False,
                   rm_target_sc = False,
                   
                   # if binder_chain is defined
                   binder_chain=None,
                   rm_binder=True,
                   rm_binder_seq=True,
                   rm_binder_sc=True,
                   rm_template_ic=False,

                   hotspot=None, ignore_missing=True,
                   target_msa=None, target_msa_mode="env", msa_depth=128, **kwargs):
    '''
    prep inputs for binder design
    ---------------------------------------------------
    -binder_len = length of binder to hallucinate (option ignored if binder_chain is defined)
    -binder_chain = chain of binder to redesign
    -use_binder_template = use binder coordinates as template input
    -rm_template_ic = use target and binder coordinates as seperate template inputs
    -hotspot = define position/hotspots on target
    -rm_[binder/target]_seq = remove sequence info from template
    -rm_[binder/target]_sc  = remove sidechain info from template
    -ignore_missing=True - skip positions that have missing density (no CA coordinate)
    -target_msa = path (or raw string) to an a3m alignment for the target chain,
                  or "auto" to fetch one automatically (single-chain target only)
                  from the public ColabFold MMseqs2 API (needs internet access).
                  When provided, the target side of the MSA feature is filled with
                  real aligned sequences (up to msa_depth rows) instead of a single
                  broadcast copy of the target sequence. This restores evolutionary/
                  co-variation signal for the target when rm_target=True removes the
                  template (i.e. template-free binder design against e.g. an antibody).
                  A manually-provided a3m must be built from the exact resolved
                  target sequence used here (see prep_msa/ignore_missing).
                  NOTE: this is independent of num_seq (which sets how many
                  *independent binder designs* are batched together, via
                  mk_af_model(num_seq=...)) -- target_msa is only supported
                  together with the default num_seq=1 (a single binder design).
    -target_msa_mode = "env" or "all", passed to fetch_msa() when target_msa="auto"
    -msa_depth = number of target_msa rows to use (default 128)
    ---------------------------------------------------
    '''
    redesign = binder_chain is not None
    rm_binder = not kwargs.pop("use_binder_template", not rm_binder)
    
    self._args.update({"redesign":redesign})

    # get pdb info
    target_chain = kwargs.pop("chain",target_chain) # backward comp
    chains = f"{target_chain},{binder_chain}" if redesign else target_chain
    im = [True] * len(target_chain.split(",")) 
    if redesign: im += [ignore_missing] * len(binder_chain.split(","))

    self._pdb = prep_pdb(pdb_filename, chain=chains, ignore_missing=im)
    res_idx = self._pdb["residue_index"]

    if redesign:
      self._target_len = sum([(self._pdb["idx"]["chain"] == c).sum() for c in target_chain.split(",")])
      self._binder_len = sum([(self._pdb["idx"]["chain"] == c).sum() for c in binder_chain.split(",")])
    else:
      self._target_len = self._pdb["residue_index"].shape[0]
      self._binder_len = binder_len
      res_idx = np.append(res_idx, res_idx[-1] + np.arange(binder_len) + 50)
    
    self._len = self._binder_len
    self._lengths = [self._target_len, self._binder_len]

    # gather hotspot info
    if hotspot is not None:
      self.opt["hotspot"] = prep_pos(hotspot, **self._pdb["idx"])["pos"]

    if redesign:
      # binder redesign
      self._wt_aatype = self._pdb["batch"]["aatype"][self._target_len:]
      self.opt["weights"].update({"dgram_cce":1.0, "rmsd":0.0, "fape":0.0,
                                  "con":0.0, "i_con":0.0, "i_pae":0.0})
    else:
      # binder hallucination
      self._pdb["batch"] = make_fixed_size(self._pdb["batch"], num_res=sum(self._lengths))
      self.opt["weights"].update({"plddt":0.1, "con":0.0, "i_con":1.0, "i_pae":0.0})

    # target MSA support (real evolutionary info for the target, instead of a
    # single broadcast copy) -- used to recover signal lost when rm_target=True
    # removes the structural template. decoupled from num_seq/self._num, which
    # instead controls how many *independent binder designs* get batched
    # together (see shared/model.py:set_seq) -- conflating the two would
    # silently turn a single binder design into N independently-randomized ones
    self._args["use_target_msa"] = target_msa is not None
    num_seq = msa_depth if target_msa is not None else 1
    if target_msa is not None:
      assert self._num == 1, (
        f"target_msa requires num_seq=1 (a single binder design), got "
        f"num_seq={self._num}. num_seq batches independent binder designs "
        f"and is unrelated to target MSA depth -- use msa_depth instead.")
      query_aatype = self._pdb["batch"]["aatype"][:self._target_len]
      if target_msa == "auto":
        assert "," not in target_chain, (
          "target_msa=\"auto\" only supports a single-chain target (no MSA "
          "pairing across chains) -- generate a paired a3m yourself via "
          "ColabFold for multi-chain targets and pass its path as target_msa.")
        query_seq = "".join(residue_constants.restypes[a] if a < 20 else "X"
                             for a in query_aatype.tolist())
        target_msa = fetch_msa(query_seq, mode=target_msa_mode)
      msa_aatype = prep_msa(target_msa, num_seq=num_seq, query_aatype=query_aatype)

    # configure input features
    self._inputs = self._prep_features(num_res=sum(self._lengths), num_seq=num_seq)
    self._inputs["residue_index"] = res_idx
    self._inputs["batch"] = self._pdb["batch"]
    if target_msa is not None:
      self._inputs["batch"]["msa_aatype"] = msa_aatype
    self._inputs.update(get_multi_id(self._lengths))

    # configure template rm masks
    (T,L,rm) = (self._lengths[0],sum(self._lengths),{})
    rm_opt = {
              "rm_template":    {"target":rm_target,    "binder":rm_binder},
              "rm_template_seq":{"target":rm_target_seq,"binder":rm_binder_seq},
              "rm_template_sc": {"target":rm_target_sc, "binder":rm_binder_sc}
             }
    for n,x in rm_opt.items():
      rm[n] = np.full(L,False)
      for m,y in x.items():
        if isinstance(y,str):
          rm[n][prep_pos(y,**self._pdb["idx"])["pos"]] = True
        else:
          if m == "target": rm[n][:T] = y
          if m == "binder": rm[n][T:] = y
        
    # set template [opt]ions
    self.opt["template"]["rm_ic"] = rm_template_ic
    self._inputs.update(rm)

    self._prep_model(**kwargs)

  def _prep_partial(self, pdb_filename, chain=None, length=None,
                    copies=1, repeat=False, homooligomer=False,
                    pos=None, fix_pos=None, use_sidechains=False, atoms_to_exclude=None,
                    rm_template=False,
                    rm_template_seq=False,
                    rm_template_sc=False,
                    rm_template_ic=False, 
                    ignore_missing=True, **kwargs):
    '''
    prep input for partial hallucination
    ---------------------------------------------------
    -length=100 - total length of protein (if different from input PDB)
    -pos="1,2-10" - specify which positions to apply supervised loss to
    -use_sidechains=True - add a sidechain supervised loss to the specified positions
      -atoms_to_exclude=["N","C","O"] (for sc_rmsd loss, specify which atoms to exclude)
    -rm_template_seq - if template is defined, remove information about template sequence
    -ignore_missing=True - skip positions that have missing density (no CA coordinate)
    ---------------------------------------------------    
    '''    
    # prep features
    self._pdb = prep_pdb(pdb_filename, chain=chain, ignore_missing=ignore_missing,
                   offsets=kwargs.pop("pdb_offsets",None),
                   lengths=kwargs.pop("pdb_lengths",None))

    self._pdb["len"] = sum(self._pdb["lengths"])

    self._len = self._pdb["len"] if length is None else length
    self._lengths = [self._len]

    # feat dims
    num_seq = self._num
    res_idx = np.arange(self._len)
    
    # get [pos]itions of interests
    if pos is None:
      self.opt["pos"] = self._pdb["pos"] = np.arange(self._pdb["len"])
      self._pos_info = {"length":np.array([self._pdb["len"]]), "pos":self._pdb["pos"]}    
    else:
      self._pos_info = prep_pos(pos, **self._pdb["idx"])
      self.opt["pos"] = self._pdb["pos"] = self._pos_info["pos"]

    if homooligomer and chain is not None and copies == 1:
      copies = len(chain.split(","))

    # repeat/homo-oligomeric support
    if copies > 1:
      
      if repeat or homooligomer:
        self._len = self._len // copies
        self._pdb["len"] = self._pdb["len"] // copies
        self.opt["pos"] = self._pdb["pos"][self._pdb["pos"] < self._pdb["len"]]

        # repeat positions across copies
        self._pdb["pos"] = repeat_pos(self.opt["pos"], copies, self._pdb["len"])

      if repeat:
        self._lengths = [self._len * copies]
        block_diag = False

      else:
        self._lengths = [self._len] * copies
        block_diag = not self._args["use_multimer"]

        num_seq = (self._num * copies + 1) if block_diag else self._num
        res_idx = repeat_idx(np.arange(self._len), copies)

        self.opt["weights"].update({"i_pae":0.0, "i_con":1.0})

      self._args.update({"copies":copies, "repeat":repeat, "homooligomer":homooligomer, "block_diag":block_diag})
      homooligomer = not repeat

    # configure input features
    self._inputs = self._prep_features(num_res=sum(self._lengths), num_seq=num_seq)
    self._inputs["residue_index"] = res_idx
    self._inputs["batch"] = jax.tree_util.tree_map(lambda x:x[self._pdb["pos"]], self._pdb["batch"])     
    self._inputs.update(get_multi_id(self._lengths, homooligomer=homooligomer))

    # configure options/weights
    self.opt["weights"].update({"dgram_cce":1.0, "rmsd":0.0, "fape":0.0, "con":1.0}) 
    self._wt_aatype = self._pdb["batch"]["aatype"][self.opt["pos"]]

    # configure sidechains
    self._args["use_sidechains"] = use_sidechains
    if use_sidechains:
      self._sc = {"batch":prep_inputs.make_atom14_positions(self._inputs["batch"]),
                  "pos":get_sc_pos(self._wt_aatype, atoms_to_exclude)}
      self.opt["weights"].update({"sc_rmsd":0.1, "sc_fape":0.1})
      self.opt["fix_pos"] = np.arange(self.opt["pos"].shape[0])      
      self._wt_aatype_sub = self._wt_aatype
      
    elif fix_pos is not None and fix_pos != "":
      sub_fix_pos = []
      sub_i = []
      pos = self.opt["pos"].tolist()
      for i in prep_pos(fix_pos, **self._pdb["idx"])["pos"]:
        if i in pos:
          sub_i.append(i)
          sub_fix_pos.append(pos.index(i))
      self.opt["fix_pos"] = np.array(sub_fix_pos)
      self._wt_aatype_sub = self._pdb["batch"]["aatype"][sub_i]
      
    elif kwargs.pop("fix_seq",False):
      self.opt["fix_pos"] = np.arange(self.opt["pos"].shape[0])
      self._wt_aatype_sub = self._wt_aatype

    self.opt["template"].update({"rm_ic":rm_template_ic})
    self._inputs.update({"rm_template":     rm_template,
                         "rm_template_seq": rm_template_seq,
                         "rm_template_sc":  rm_template_sc})
  
    self._prep_model(**kwargs)

#######################
# utils
#######################
def repeat_idx(idx, copies=1, offset=50):
  idx_offset = np.repeat(np.cumsum([0]+[idx[-1]+offset]*(copies-1)),len(idx))
  return np.tile(idx,copies) + idx_offset

def repeat_pos(pos, copies, length):
  return (np.repeat(pos,copies).reshape(-1,copies) + np.arange(copies) * length).T.flatten()

def _align_cols(a, b):
  '''
  align sequence [a] (e.g. the resolved/target sequence) onto sequence [b]
  (e.g. an a3m's own query row, usually the full expressed sequence).
  returns a list of length len(a): for each position in a, the matching
  index into b, or None if no match was found (e.g. missing density in a
  that isn't simply an unaligned insertion in b).
  '''
  import difflib
  sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
  cols = [None] * len(a)
  for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag in ("equal", "replace"):
      n = min(i2 - i1, j2 - j1)
      for k in range(n):
        cols[i1 + k] = j1 + k
  return cols

def prep_msa(a3m, num_seq, query_aatype=None):
  '''
  parse an a3m alignment (file path or raw string) into an aatype array
  of shape (num_seq, L), for use as the "target_msa" input of _prep_binder.
  -query_aatype - if given, the alignment is mapped onto this exact target
                  sequence (so the row AlphaFold treats as the query always
                  matches what's used elsewhere in prep). If the a3m's own
                  query row differs in length (e.g. it was built from the
                  full expressed antibody sequence, while ignore_missing
                  dropped some unresolved loop residues from the structure),
                  columns are aligned by sequence and any target residue
                  that can't be aligned is gap-filled (no MSA info there).
  '''
  import os
  a3m_string = open(a3m).read() if os.path.isfile(a3m) else a3m
  # strip ColabFold's paired-MSA cardinality header ("#len1,len2\t1,1"), which
  # parse_a3m (a plain a3m parser) doesn't expect
  a3m_string = "\n".join(l for l in a3m_string.splitlines() if not l.startswith("#"))
  seqs, _ = parsers.parse_a3m(a3m_string)

  aa_to_id = residue_constants.HHBLITS_AA_TO_ID
  mapping = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
  msa = np.array([[mapping[aa_to_id.get(a.upper(),20)] for a in s] for s in seqs])

  if query_aatype is not None:
    query_aatype = np.asarray(query_aatype)
    target_seq = "".join(residue_constants.restypes[a] if a < 20 else "X"
                          for a in query_aatype.tolist())
    if msa.shape[1] != len(target_seq) or seqs[0] != target_seq:
      cols = _align_cols(target_seq, seqs[0])
      n_missing = cols.count(None)
      if n_missing > 0:
        print(f"NOTE: {n_missing}/{len(cols)} target residues could not be "
              f"aligned to the target_msa and will have no MSA information "
              f"(gap-filled) -- this is expected if those residues have "
              f"missing density in the input structure.")
      new_msa = np.full((msa.shape[0], len(cols)), 21, dtype=msa.dtype)
      for i, c in enumerate(cols):
        if c is not None:
          new_msa[:, i] = msa[:, c]
      msa = new_msa
    msa[0] = query_aatype

  # select/pad to the requested number of rows (query row always kept)
  n = msa.shape[0]
  if n >= num_seq:
    msa = msa[:num_seq]
  else:
    pad = np.full((num_seq - n, msa.shape[1]), 21, dtype=msa.dtype) # pad with gap
    msa = np.concatenate([msa, pad], 0)
  return msa

def fetch_msa(sequence, mode="env", host_url="https://api.colabfold.com",
              user_agent="colabdesign"):
  '''
  fetch an a3m alignment for a single-chain [sequence] from the public
  ColabFold MMseqs2 API -- the same server ColabFold's own notebooks use
  to build MSAs. Returns the a3m contents as a string.
  -mode = "env" (MGnify+ColabFoldDB, matches ColabFold's default) or
          "all" (+ UniRef, slower/deeper)
  Note: this only searches a single sequence. For multi-chain/paired
  complexes (e.g. an antibody Fab), pairing needs ColabFold's own
  search pipeline (colabfold_search / the ColabFold notebook) -- generate
  the a3m there and pass its path as target_msa instead.
  '''
  import json, tarfile, io, time
  import urllib.request, urllib.parse

  headers = {"User-Agent": user_agent}

  def _request(url, data=None):
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
      return r.read()

  query = f">query\n{sequence}\n"
  data = urllib.parse.urlencode({"q":query, "mode":mode}).encode()
  res = json.loads(_request(f"{host_url}/ticket/msa", data=data))
  ticket_id = res["id"]

  status = res.get("status","")
  while status not in ("COMPLETE","ERROR"):
    time.sleep(5)
    res = json.loads(_request(f"{host_url}/ticket/{ticket_id}"))
    status = res.get("status","")

  if status == "ERROR":
    raise RuntimeError(
      "the ColabFold MMseqs2 API returned an error while computing the MSA "
      "-- try again later, or generate the a3m yourself via ColabFold.")

  tar_bytes = _request(f"{host_url}/result/download/{ticket_id}")
  a3m_lines = []
  with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
    for name in sorted(tar.getnames()):
      if name.endswith(".a3m"):
        f = tar.extractfile(name)
        if f is not None:
          a3m_lines.append(f.read().decode())
  if len(a3m_lines) == 0:
    raise RuntimeError("no .a3m file found in the MMseqs2 API result")
  return "\n".join(a3m_lines)

def prep_pdb(pdb_filename, chain=None,
             offsets=None, lengths=None,
             ignore_missing=False):
  '''extract features from pdb'''

  def add_cb(batch):
    '''add missing CB atoms based on N,CA,C'''
    p,m = batch["all_atom_positions"], batch["all_atom_mask"]
    atom_idx = residue_constants.atom_order
    atoms = {k:p[...,atom_idx[k],:] for k in ["N","CA","C"]}
    cb = atom_idx["CB"]
    cb_atoms = _np_get_cb(**atoms, use_jax=False)
    cb_mask = np.prod([m[...,atom_idx[k]] for k in ["N","CA","C"]],0)
    batch["all_atom_positions"][...,cb,:] = np.where(m[:,cb,None], p[:,cb,:], cb_atoms)
    batch["all_atom_mask"][...,cb] = (m[:,cb] + cb_mask) > 0
    return {"atoms":batch["all_atom_positions"][:,cb],"mask":cb_mask}

  if isinstance(chain,str) and "," in chain:
    chains = chain.split(",")
  elif not isinstance(chain,list):
    chains = [chain]

  o,last = [],0
  residue_idx, chain_idx = [],[]
  full_lengths = []

  # go through each defined chain  
  for n,chain in enumerate(chains):
    pdb_str = pdb_to_string(pdb_filename, chains=chain, models=[1])
    protein_obj = protein.from_pdb_string(pdb_str, chain_id=chain)
    batch = {'aatype': protein_obj.aatype,
             'all_atom_positions': protein_obj.atom_positions,
             'all_atom_mask': protein_obj.atom_mask,
             'residue_index': protein_obj.residue_index}

    cb_feat = add_cb(batch) # add in missing cb (in the case of glycine)
    
    im = ignore_missing[n] if isinstance(ignore_missing,list) else ignore_missing
    if im:
      r = batch["all_atom_mask"][:,0] == 1
      batch = jax.tree_util.tree_map(lambda x:x[r], batch)
      residue_index = batch["residue_index"] + last

    else:
      # pad values
      offset = 0 if offsets is None else (offsets[n] if isinstance(offsets,list) else offsets)
      r = offset + (protein_obj.residue_index - protein_obj.residue_index.min())
      length = (r.max()+1) if lengths is None else (lengths[n] if isinstance(lengths,list) else lengths)    
      def scatter(x, value=0):
        shape = (length,) + x.shape[1:]
        y = np.full(shape, value, dtype=x.dtype)
        y[r] = x
        return y

      batch = {"aatype":scatter(batch["aatype"],-1),
               "all_atom_positions":scatter(batch["all_atom_positions"]),
               "all_atom_mask":scatter(batch["all_atom_mask"]),
               "residue_index":scatter(batch["residue_index"],-1)}
      
      residue_index = np.arange(length) + last
    
    last = residue_index[-1] + 50
    o.append({"batch":batch,
              "residue_index": residue_index,
              "cb_feat":cb_feat})
    
    residue_idx.append(batch.pop("residue_index"))
    chain_idx.append([chain] * len(residue_idx[-1]))
    full_lengths.append(len(residue_index))

  # concatenate chains
  o = jax.tree_util.tree_map(lambda *x:np.concatenate(x,0),*o)
  
  # save original residue and chain index
  o["idx"] = {"residue":np.concatenate(residue_idx), "chain":np.concatenate(chain_idx)}
  o["lengths"] = full_lengths
  return o

def make_fixed_size(feat, num_res, num_seq=1, num_templates=1):
  '''pad input features'''
  shape_schema = {k:v for k,v in config.CONFIG.data.eval.feat.items()}

  pad_size_map = {
      shape_placeholders.NUM_RES: num_res,
      shape_placeholders.NUM_MSA_SEQ: num_seq,
      shape_placeholders.NUM_EXTRA_SEQ: 1,
      shape_placeholders.NUM_TEMPLATES: num_templates
  }  
  for k,v in feat.items():
    if k == "batch":
      feat[k] = make_fixed_size(v, num_res)
    else:
      shape = list(v.shape)
      schema = shape_schema[k]
      assert len(shape) == len(schema), (
          f'Rank mismatch between shape and shape schema for {k}: '
          f'{shape} vs {schema}')
      pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
      padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
      feat[k] = np.pad(v, padding)
  return feat

def get_sc_pos(aa_ident, atoms_to_exclude=None):
  '''get sidechain indices/weights for all_atom14_positions'''

  # decide what atoms to exclude for each residue type
  a2e = {}
  for r in resname_to_idx:
    if isinstance(atoms_to_exclude,dict):
      a2e[r] = atoms_to_exclude.get(r,atoms_to_exclude.get("ALL",["N","C","O"]))
    else:
      a2e[r] = ["N","C","O"] if atoms_to_exclude is None else atoms_to_exclude

  # collect atom indices
  pos,pos_alt = [],[]
  N,N_non_amb = [],[]
  for n,a in enumerate(aa_ident):
    aa = idx_to_resname[a]
    atoms = set(residue_constants.residue_atoms[aa])
    atoms14 = residue_constants.restype_name_to_atom14_names[aa]
    swaps = residue_constants.residue_atom_renaming_swaps.get(aa,{})
    swaps.update({v:k for k,v in swaps.items()})
    for atom in atoms.difference(a2e[aa]):
      pos.append(n * 14 + atoms14.index(atom))
      if atom in swaps:
        pos_alt.append(n * 14 + atoms14.index(swaps[atom]))
      else:
        pos_alt.append(pos[-1])
        N_non_amb.append(n)
      N.append(n)

  pos, pos_alt = np.asarray(pos), np.asarray(pos_alt)
  non_amb = pos == pos_alt
  N, N_non_amb = np.asarray(N), np.asarray(N_non_amb)
  w = np.array([1/(n == N).sum() for n in N])
  w_na = np.array([1/(n == N_non_amb).sum() for n in N_non_amb])
  w, w_na = w/w.sum(), w_na/w_na.sum()
  return {"pos":pos, "pos_alt":pos_alt, "non_amb":non_amb,
          "weight":w, "weight_non_amb":w_na[:,None]}

def prep_input_features(L, N=1, T=1, eN=1):
  '''
  given [L]ength, [N]umber of sequences and number of [T]emplates
  return dictionary of blank features
  '''
  inputs = {'aatype': np.zeros(L,int),
            'target_feat': np.zeros((L,20)),
            'msa_feat': np.zeros((N,L,49)),
            # 23 = one_hot -> (20, UNK, GAP, MASK)
            # 1  = has deletion
            # 1  = deletion_value
            # 23 = profile
            # 1  = deletion_mean_value
  
            'seq_mask': np.ones(L),
            'msa_mask': np.ones((N,L)),
            'msa_row_mask': np.ones(N),
            'atom14_atom_exists': np.zeros((L,14)),
            'atom37_atom_exists': np.zeros((L,37)),
            'residx_atom14_to_atom37': np.zeros((L,14),int),
            'residx_atom37_to_atom14': np.zeros((L,37),int),            
            'residue_index': np.arange(L),
            'extra_deletion_value': np.zeros((eN,L)),
            'extra_has_deletion': np.zeros((eN,L)),
            'extra_msa': np.zeros((eN,L),int),
            'extra_msa_mask': np.zeros((eN,L)),
            'extra_msa_row_mask': np.zeros(eN),

            # for template inputs
            'template_aatype': np.zeros((T,L),int),
            'template_all_atom_mask': np.zeros((T,L,37)),
            'template_all_atom_positions': np.zeros((T,L,37,3)),
            'template_mask': np.zeros(T),
            'template_pseudo_beta': np.zeros((T,L,3)),
            'template_pseudo_beta_mask': np.zeros((T,L)),

            # for alphafold-multimer
            'asym_id': np.zeros(L),
            'sym_id': np.zeros(L),
            'entity_id': np.zeros(L),
            'all_atom_positions': np.zeros((N,37,3))}
  return inputs

def get_multi_id(lengths, homooligomer=False):
  '''set info for alphafold-multimer'''
  i = np.concatenate([[n]*l for n,l in enumerate(lengths)])
  if homooligomer:
    return {"asym_id":i, "sym_id":i, "entity_id":np.zeros_like(i)}
  else:
    return {"asym_id":i, "sym_id":i, "entity_id":i}