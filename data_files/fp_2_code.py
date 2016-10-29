#!/usr/bin/python
# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: convert mgfp(morgan fingerprint) file into binary_code file

import os
import sys
import h5py
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def read_fp(filename):
  """ read fingerprint from file
  Args:
    filename: <type 'str'>
  Return:
    chembl_id_list: <type 'list'>, a list of str
    fps_list: <type 'list'>, a list of dict.
  """
  chembl_id_list = []
  fps_list = []

  infile = open(filename, "r")
  line_num = 0
  for line in infile:
    line_num += 1
    chembl_id = line.split("\t")[0].strip()
    fps_str = line.split("\t")[1].strip()
    fps = {}
    fps_str = fps_str[1:-1].split(",")
    for fp in fps_str:
      if ":" in fp:
        k, v = fp.split(":")
        k = int(k)
        v = int(v)
        assert k not in fps.keys(), ("error in fp_file %s at line %d: dict's keys duplicated" % (filename, line_num))
        fps[k] = v 
    chembl_id_list.append(chembl_id)
    fps_list.append(fps)

  infile.close()
  return chembl_id_list, fps_list


#def fp2mat(fps_list, vec_len=1021, hash_func=None, dtype=np.int16):
def fp2mat(fps_list, vec_len=2039, hash_func=None, dtype=np.int16):
  """ convert fingerprint(for short: fp) into a csr_matrix

  Args:
    fps_list: <type 'list'>, list of dict, such as [{5534: 1, 78976: 34}],
        contains the fingerprint of a molecule, 
        where the keys are the id of a specifical fingerprint, 
        and the values are the number that the molecule contains the fingerprint.
    vec_len: <type 'int'>, default value is 1021 or 2039, a prime number is prefered.
        !!! Warning: never use vec_len=2048, when hash_func=None.
        the length of the vector to be returned.
    hash_func: <type 'function'>, default is None,
        the function which hashes fp id into a integer number.
        Note that the hashed number should be less than vec_len.
        if None, simply devide fp id with vec_len and return remainder, 
        i.e. hashed number = fp id % vec_len, or hashed number = mod(fp id, vec_len)
    dtype: <type 'type'>, default is numpy.int16

  Return:
    mat: <calss 'scipy.sparse.csr.csr_matrix'>,
        the shape of vec must be (len(fps_list), vec_len)
  """
  if hash_func == None:
    hash_func = lambda x: x % vec_len

  indptr = [0]
  indices = []
  data = []
  for fps in fps_list:
    for (k, v) in fps.items():
      indices.append(hash_func(k))
      data.append(v)
    indptr.append(len(indices))

  a = sparse.csr_matrix((data, indices, indptr), shape=(len(fps_list), vec_len), dtype=dtype)

  # check whether too much infomation was missed after hash.
  # original info: np.array([len(x) for x in fps_list]).mean()
  # after hash info: (a.toarray() != 0).sum(axis=1).mean()
  #print(a.shape, a.max(), a.min(), (a.toarray() != 0).sum(axis=1).mean(), np.array([len(x) for x in fps_list]).mean())

  return a
       



if __name__ == "__main__":


  fp_dir = "fp_files"
  h5_dir = "h5_files"
  if not os.path.exists(h5_dir):
    os.mkdir(h5_dir)


  # 8 target
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  #target_list.append("pubchem_neg_sample")

  """
  for target in target_list:
    print("process %s..." % target)

    # open an h5 file to save one target's sparse matrix of fingerprint
    h5_fn = os.path.join(h5_dir, target + ".h5")
    h5 = h5py.File(h5_fn, "w")
  
    # read mgfp from file and encode into sparse matrix
    mgfp_fn = os.path.join(fp_dir, target + ".mgfp6")
    id_list, mgfps_list = read_fp(mgfp_fn)
    mgmat = fp2mat(mgfps_list)

    # save chembl id
    h5["chembl_id"] = id_list

    # save mgfp's sparse matrix
    mgh5 = h5.create_group("/mg")
    mgh5["data"] = mgmat.data
    mgh5["indices"] = mgmat.indices
    mgh5["indptr"] = mgmat.indptr

    # read apfp from file and encode into sparse matrix
    apfp_fn = os.path.join(fp_dir, target + ".apfp")
    _, apfps_list = read_fp(apfp_fn)
    apmat = fp2mat(apfps_list)

    # save apfp's sparse matrix
    aph5 = h5.create_group("/ap")
    aph5["data"] = apmat.data
    aph5["indices"] = apmat.indices
    aph5["indptr"] = apmat.indptr

    # read ttfp from file and encode into sparse matrix
    ttfp_fn = os.path.join(fp_dir, target + ".ttfp")
    _, ttfps_list = read_fp(ttfp_fn)
    ttmat = fp2mat(ttfps_list)

    # save ttfp's sparse matrix
    tth5 = h5.create_group("/tt")
    tth5["data"] = ttmat.data
    tth5["indices"] = ttmat.indices
    tth5["indptr"] = ttmat.indptr

    h5.close()
  """

  """ add label
  for target in target_list:
    print("add label year for %s" % target)
    # read label
    mgfp_fn = os.path.join(fp_dir, target + ".mgfp6")
    infile = open(mgfp_fn, "r")
    label_list = []
    year_list = []
    for line in infile:
      line_ = line.split("\t")
      label = int(line_[2])
      if line_[3].isdigit():
        print(line_[3])
        year = int(line_[3]) 
      else:
        year = -1 
      label_list.append(label)
      year_list.append(year)
    infile.close()

    # write label into h5 file
    h5_fn = os.path.join(h5_dir, target + ".h5")
    h5 = h5py.File(h5_fn, "r+")
    if "label" in h5.keys():
      del h5["label"]
    h5["label"] = np.array(label_list)
    if "year" in h5.keys():
      del h5["year"]
    h5["year"] = np.array(year_list)
    h5.close()
  """

  """

  part_num = int(sys.argv[1])

  # pubchem all compounds
  mgfp_dir = "/raid/xiaotaw/pubchem/morgan_fp"
  pkl_dir = "/raid/xiaotaw/pubchem/pkl_files"


  #for i in xrange(1, 121225001, 25000):
  begin_num = part_num * 10000000 + 1
  if part_num == 11:
    end_num = 121225001
  else:
    end_num = (part_num + 1) * 10000000 + 1  

  for i in xrange(begin_num, end_num, 25000):
    in_file = "Compound_" + "{:0>9}".format(i) + "_" + "{:0>9}".format(i + 24999) + ".mgfp"
    if not os.path.exists(os.path.join(mgfp_dir, in_file)):
      print("%s\t0\tnot exists" % in_file)
      continue
    mgfp2code(in_file, 
              has_label=False, add_label=False, 
              mgfp_dir=mgfp_dir, pkl_dir=pkl_dir, code_len=8192)

  """


#""" generate mask for pns (generate cliped_mask)
pns_fn = "h5_files/pubchem_neg_sample.h5"
pns_hf = h5py.File(pns_fn, "r+")
#pns_m = pns_hf.create_group("mask")
pns_m = pns_hf.create_group("cliped_mask")

vec_len = 2039

ap = sparse.csr_matrix((pns_hf["ap"]["data"], pns_hf["ap"]["indices"], pns_hf["ap"]["indptr"]), shape=[len(pns_hf["ap"]["indptr"]) - 1, vec_len])
mg = sparse.csr_matrix((pns_hf["mg"]["data"], pns_hf["mg"]["indices"], pns_hf["mg"]["indptr"]), shape=[len(pns_hf["mg"]["indptr"]) - 1, vec_len])
tt = sparse.csr_matrix((pns_hf["tt"]["data"], pns_hf["tt"]["indices"], pns_hf["tt"]["indptr"]), shape=[len(pns_hf["tt"]["indptr"]) - 1, vec_len])

#pns = sparse.hstack([ap, mg, tt]).toarray()
pns = np.clip(sparse.hstack([ap, mg, tt]).toarray(), 0, 1)



target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

h5_dir = "h5_files"
for target in target_list:
  h5_fn = os.path.join(h5_dir, target + ".h5")
  hf = h5py.File(h5_fn, "r+")
  if "mask" in hf.keys():
    del hf["mask"]
  ap = sparse.csr_matrix((hf["ap"]["data"], hf["ap"]["indices"], hf["ap"]["indptr"]), shape=[len(hf["ap"]["indptr"]) - 1, vec_len])
  mg = sparse.csr_matrix((hf["mg"]["data"], hf["mg"]["indices"], hf["mg"]["indptr"]), shape=[len(hf["mg"]["indptr"]) - 1, vec_len])
  tt = sparse.csr_matrix((hf["tt"]["data"], hf["tt"]["indices"], hf["tt"]["indptr"]), shape=[len(hf["tt"]["indptr"]) - 1, vec_len])
  features = np.clip(sparse.hstack([ap, mg, tt]).toarray(), 0, 1)
  a = features[hf["label"].value.astype(bool)]
  mask = []
  for l in pns:
    if (np.square(l - a).sum(axis=1) == 0).sum() != 0:
      mask.append(True)
    else:
      mask.append(False)
  mask = np.array(mask)
  pns_m[target] = mask
  print("%s\t%d" % (target, mask.sum()))
  hf.close()



pns_hf.close()
#"""













