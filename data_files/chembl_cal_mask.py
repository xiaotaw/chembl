# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Dec 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description: calculate mask(label) of chembl molecules for specific targets

import os
import sys
import math
import time
import datetime
import multiprocessing
import numpy as np
from scipy import sparse
from collections import defaultdict

# folders
fp_dir = "fp_files"
structure_dir = "structure_files"
mask_dir = "mask_files"
if not os.path.exists(mask_dir):
  os.mkdir(mask_dir)
log_dir = "log_files"
if not os.path.exists(log_dir):
  os.mkdir(log_dir)


# the newly picked out 15 targets, include 9 targets from 5 big group, and 6 targets from others.
target_list = ["CHEMBL279", "CHEMBL203", # Protein Kinases
               "CHEMBL217", "CHEMBL253", # GPCRs (Family A)
               "CHEMBL235", "CHEMBL206", # Nuclear Hormone Receptors
               "CHEMBL240", "CHEMBL4296", # Voltage Gated Ion Channels
               "CHEMBL4805", # Ligand Gated Ion Channels
               "CHEMBL204", "CHEMBL244", "CHEMBL4822", "CHEMBL340", "CHEMBL205", "CHEMBL4005" # Others
              ] 

# the target
#target = target_list[int(sys.argv[1])]

# read chembl id and apfp
chembl_id = []
chembl_apfp = {}
f = open(os.path.join(fp_dir, "chembl.apfp"), "r")
for line in f:
  id_, fps_str = line.split("\t")
  id_ = id_.strip()
  fps_str = fps_str.strip()
  chembl_id.append(id_)
  chembl_apfp[id_] = fps_str

f.close()

# read (pubchem negative sample)pns apfp and counts the fps that appeared in pns compounds
pns_id = []
pns_apfp = {}
pns_count = defaultdict(lambda : 0)
f = open(os.path.join(fp_dir, "pubchem_neg_sample.apfp"), "r")
for line in f:
  id_, fps_str = line.split("\t")
  id_ = id_.strip()
  fps_str = fps_str.strip()
  pns_id.append(id_)
  pns_apfp[id_] = fps_str
  for fp in fps_str[1:-1].split(","):
    if ":" in fp:
      k, _ = fp.split(":")
      pns_count[int(k)] += 1

f.close()


# read top 79 targets' label
clf_label_79 = np.genfromtxt(os.path.join(structure_dir, "chembl_top79.label"), usecols=[0, 2, 3], delimiter="\t", skip_header=1, dtype=str)

def cal_mask(target):
  ################################################################################
  # generate sparse matrix for target features 

  # target compounds' chembl_id and clf label.
  target_clf_label = clf_label_79[clf_label_79[:, 0] == target]

  # remove compounds whose apfp cannot be caculated
  m = []
  for cmpd_id in target_clf_label[:, 1]:
    if cmpd_id in chembl_id:
      m.append(True)
    else:
      m.append(False)
  target_clf_label = target_clf_label[np.array(m)]  

  # target fps
  target_fps = [chembl_apfp[x] for x in target_clf_label[:, 1]]

  # count the fps that appeared in the compounds of the target
  target_count = defaultdict(lambda : 0)
  for fps_str in target_fps:
    for fp in fps_str[1:-1].split(","):
      if ":" in fp:
        k, _ = fp.split(":")
        target_count[int(k)] += 1

  target_count.update(pns_count)

  # save target apfp count 
  count_file = open(os.path.join(mask_dir, "%s_apfp.count" % target), "w")
  for k in target_count.keys():
    count_file.write("%d\t%d\n" % (k, target_count[k]))

  count_file.close()
    
  # pick out that fps that appeared for more than 10 times.
  # Here we assume that the more frequently a fp appeared, the more important it is.
  v = np.array([[k, target_count[k]] for k in target_count.keys()])
  m = v[:, 1] > 10
  target_apfp_picked = v[m][:, 0]
  
  # according to the apfp that picked out, define the columns in the feature sparse matrix
  # Note: a defaultdict is used. 
  # And the purpose is assign a default value(length of target_apfp_picked) for the apfps 
  # which is not included in target_apfp_picked. And this column(the last column) was finally 
  # not used at all.
  columns_dict = defaultdict(lambda : len(target_apfp_picked))
  for i, apfp in enumerate(target_apfp_picked):
    columns_dict[apfp] = i
  
  # define the function which can construct a feature sparse matrix according to the columns_dict
  def sparse_features(fps_list):
    data = []
    indices = []
    indptr = [0]
    for fps_str in fps_list:
      n = indptr[-1]
      for fp in fps_str[1:-1].split(","):
        if ":" in fp:
          k, v = fp.split(":")
          indices.append(columns_dict[int(k)])
          data.append(int(v))
          n += 1
      indptr.append(n)
    a = sparse.csr_matrix((np.array(data), indices, indptr), shape=(len(fps_list), len(target_apfp_picked) + 1))
    return a
  
  # pick out target compounds with pos labels
  # normally, abs(clf_label) > 0.5(refer to chembl_preparation.py), 
  # so it also works when using the following line:
  # target_pos_id = target_clf_label[target_clf_label[:, 2].astype(float) > 0.5][:, 1]
  target_pos_id = target_clf_label[target_clf_label[:, 2].astype(float) > 0][:, 1]
  target_pos_fps = [chembl_apfp[x] for x in target_pos_id]
  
  # generate feature sparse matrix for target's pos compounds
  target_pos_features = sparse_features(target_pos_fps)[:, :-1].toarray()
  
  # generate feature sparse matrix for pns compounds
  target_pns_features = sparse_features([pns_apfp[k] for k in pns_id])[:, :-1]
  
  # generate feature sparse matrix for (chembl negative sample)cns compounds
  target_cns_features = sparse_features([chembl_apfp[k] for k in chembl_id])[:, :-1]
  
  
  ################################################################################
  # generate mask for pns and cns
  
  # define a task function for sub process:
  # it can compare a part of negative sample(cns or pns) with pos samples,
  # and return the mask of those samples back to the main process.
  def sub_compare(sub_neg_id, sub_neg_features, conn):
    mask = {}
    log_str = []
    for neg_k, neg_f in zip(sub_neg_id, sub_neg_features):
      for pos_k, pos_f in zip(target_pos_id, target_pos_features):
        if (neg_f != pos_f).sum() == 0:
          mask[neg_k] = True
          log_str.append("%s\t%s\n" % (neg_k, pos_k))
    conn.send((mask, log_str))
    conn.close()
  
  # the number of sub process for computation 
  n_jobs = 6
  
  
  # using multiprocessing compute mask for pns
  t1 = time.time()
  date1 = datetime.datetime.now()
  
  num_per_job = int(math.ceil(target_pns_features.shape[0] / float(n_jobs)))
  thread_list = []
  conn_list = []
  for i in range(0, n_jobs):
    begin = i * num_per_job
    end = (i + 1) * num_per_job
    if end > target_pns_features.shape[0]:
      end = target_pns_features.shape[0]
    p_conn, c_conn = multiprocessing.Pipe()
    conn_list.append((p_conn, c_conn))
    t = multiprocessing.Process(target=sub_compare, args=(pns_id[begin: end], target_pns_features[begin: end], c_conn))
    thread_list.append(t)
  
  for i in range(n_jobs):  
    thread_list[i].start()
  
  for i in range(n_jobs):  
    thread_list[i].join()
  
  t2 = time.time()
  
  target_pns_mask = defaultdict(lambda : False)
  
  log = open(log_dir + "/" + target + "_gen_pns_mask.log", "w")
  log.write("%s generate mask for pubchem neg sample, begins at %s\n" % (target, str(date1)))
  
  for i in range(n_jobs):  
    p_conn = conn_list[i][0]
    mask, log_str = p_conn.recv()
    target_pns_mask.update(mask)
    log.writelines(log_str)
  
  log.write("generate mask for pns, duration: %.3f\n" % (t2 - t1))
  log.close()
  
  mask_file = open(os.path.join(mask_dir, "%s_pns.mask" % target), "w")
  mask_file.writelines(["%s\t%s\n" % (x, target_pns_mask[x]) for x in pns_id])
  mask_file.close()
  
  print("generate mask for pns, duration: %.3f" % (t2 - t1))
  
  
  # using multiprocessing compute mask for cns 
  t2 = time.time()
  date2 = datetime.datetime.now()
  
  num_per_job = int(math.ceil(target_cns_features.shape[0] / float(n_jobs)))
  thread_list = []
  conn_list = []
  for i in range(0, n_jobs):
    begin = i * num_per_job
    end = (i + 1) * num_per_job
    if end > target_cns_features.shape[0]:
      end = target_cns_features.shape[0]
    p_conn, c_conn = multiprocessing.Pipe()
    conn_list.append((p_conn, c_conn))
    t = multiprocessing.Process(target=sub_compare, args=(chembl_id[begin: end], target_cns_features[begin: end], c_conn))
    thread_list.append(t)
  
  for i in range(n_jobs):  
    thread_list[i].start()
  
  for i in range(n_jobs):  
    thread_list[i].join()
  
  t3 = time.time()
  
  target_cns_mask = defaultdict(lambda : False)
  
  log = open(log_dir + "/" + target + "_gen_cns_mask.log", "w")
  log.write("%s generate mask for chembl neg sample, begins at %s\n" % (target, str(date2)))
  
  for i in range(n_jobs):  
    p_conn = conn_list[i][0]
    mask, log_str = p_conn.recv()
    target_cns_mask.update(mask)
    log.writelines(log_str)
  
  log.write("generate mask for cns, duration: %.3f\n" % (t3 - t2))
  log.close()
  
  mask_file = open(os.path.join(mask_dir, "%s_cns.mask" % target), "w")
  mask_file.writelines(["%s\t%s\n" % (x, target_cns_mask[x]) for x in chembl_id])
  mask_file.close()
  
  print("generate mask for cns, duration: %.3f" % (t3 - t2))
  

# the newly picked out 15 targets, include 9 targets from 5 big group, and 6 targets from others.
target_list = [
               "CHEMBL4805", # Ligand Gated Ion Channels
               "CHEMBL244", "CHEMBL4822", "CHEMBL340", "CHEMBL205", "CHEMBL4005" # Others
              ] 


#for target in target_list:
#  cal_mask(target)
cal_mask("CHEMBL205")
