# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Dec 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description: calculate mask(label) of chembl molecules for specific targets

import os
import math
import time
import datetime
import multiprocessing
import numpy as np
from scipy import sparse
from collections import defaultdict


target_list = ['CHEMBL203', 'CHEMBL204', 'CHEMBL205', 'CHEMBL214', 'CHEMBL217',
       'CHEMBL218', 'CHEMBL220', 'CHEMBL224', 'CHEMBL225', 'CHEMBL226',
       'CHEMBL228', 'CHEMBL230', 'CHEMBL233', 'CHEMBL234', 'CHEMBL235',
       'CHEMBL236', 'CHEMBL237', 'CHEMBL240', 'CHEMBL244', 'CHEMBL251',
       'CHEMBL253', 'CHEMBL256', 'CHEMBL259', 'CHEMBL260', 'CHEMBL261',
       'CHEMBL264', 'CHEMBL267', 'CHEMBL279', 'CHEMBL284', 'CHEMBL2842',
       'CHEMBL289', 'CHEMBL325', 'CHEMBL332', 'CHEMBL333', 'CHEMBL340',
       'CHEMBL344', 'CHEMBL4005', 'CHEMBL4296', 'CHEMBL4722', 'CHEMBL4822']

# read chembl id and apfp
chembl_id = []
chembl_apfp = {}
f = open("fp_files/chembl.apfp", "r")
for line in f:
  id_, fps_str = line.split("\t")
  id_ = id_.strip()
  fps_str = fps_str.strip()
  chembl_id.append(id_)
  chembl_apfp[id_] = fps_str

f.close()

# read pns apfp
pns_id = []
pns_apfp = {}
pns_count = defaultdict(lambda : 0)
f = open("fp_files/pubchem_neg_sample.apfp", "r")
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

# read top 40 targets' label
clf_label_40 = np.genfromtxt("structure_files/chembl_top40.label", usecols=[0, 2, 3], delimiter="\t", skip_header=1, dtype=str)


# generate sparse matrix for target features 
target = target_list[0]
target_clf_label = clf_label_40[clf_label_40[:, 0] == target]


target_fps = [chembl_apfp[x] for x in target_clf_label[:, 1]]

target_count = defaultdict(lambda : 0)
for fps_str in target_fps:
  for fp in fps_str[1:-1].split(","):
    if ":" in fp:
      k, _ = fp.split(":")
      target_count[int(k)] += 1

target_count.update(pns_count)
v = np.array([[k, target_count[k]] for k in target_count.keys()])
m = v[:, 1] > 10
target_apfp_picked = v[m][:, 0]


columns_dict = defaultdict(lambda : len(target_apfp_picked))
for i, apfp in enumerate(target_apfp_picked):
  columns_dict[apfp] = i

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



target_pos_id = target_clf_label[target_clf_label[:, 2].astype(float) > 0][:, 1]
target_pos_fps = [chembl_apfp[x] for x in target_pos_id]
target_pos_features = sparse_features(target_pos_fps)[:, :-1].toarray()

#target_pns_features = sparse_features([pns_apfp[k] for k in pns_id])[:, :-1]

target_chembl_features = sparse_features([chembl_apfp[k] for k in chembl_id])[:, :-1]


# generate mask for pns and cns
log_files = "log_files"
if not os.path.exists(log_files):
  os.mkdir(log_files)

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

n_jobs = 6


# for pns
"""
t1 = time.time()
date1 = datetime.datetime.now()

num_per_job = int(math.ceil(target_pns_features.shape[0] / float(n_jobs)))
#num_per_job = 100
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


log = open(log_files + "/" + target + "_gen_pns_mask.log", "w")
log.write("%s generate mask for pubchem neg sample, begins at %s\n" % (target, str(date1)))

for i in range(n_jobs):  
  p_conn = conn_list[i][0]
  mask, log_str = p_conn.recv()
  target_pns_mask.update(mask)
  log.writelines(log_str)

log.write("generate mask for pns, duration: %.3f\n" % (t2 - t1))
log.close()

mask_file = open("fp_files/%s_pns.mask" % target, "w")
mask_file.writelines(["%s\t%s\n" % (x, target_pns_mask[x]) for x in pns_id])
mask_file.close()


print("generate mask for pns, duration: %.3f" % (t2 - t1))
"""



# for cns 
t2 = time.time()
date2 = datetime.datetime.now()

num_per_job = int(math.ceil(target_chembl_features.shape[0] / float(n_jobs)))
#num_per_job = 100
thread_list = []
conn_list = []
for i in range(0, n_jobs):
  begin = i * num_per_job
  end = (i + 1) * num_per_job
  if end > target_chembl_features.shape[0]:
    end = target_chembl_features.shape[0]
  p_conn, c_conn = multiprocessing.Pipe()
  conn_list.append((p_conn, c_conn))
  t = multiprocessing.Process(target=sub_compare, args=(chembl_id[begin: end], target_chembl_features[begin: end], c_conn))
  thread_list.append(t)

for i in range(n_jobs):  
  thread_list[i].start()

for i in range(n_jobs):  
  thread_list[i].join()

t3 = time.time()

target_cns_mask = defaultdict(lambda : False)

log = open(log_files + "/" + target + "_gen_cns_mask.log", "w")
log.write("%s generate mask for chembl neg sample, begins at %s\n" % (target, str(date2)))

for i in range(n_jobs):  
  p_conn = conn_list[i][0]
  mask, log_str = p_conn.recv()
  target_cns_mask.update(mask)
  log.writelines(log_str)

log.write("generate mask for cns, duration: %.3f\n" % (t3 - t2))
log.close()

mask_file = open("fp_files/%s_cns.mask" % target, "w")
mask_file.writelines(["%s\t%s\n" % (x, target_cns_mask[x]) for x in chembl_id])
mask_file.close()


print("generate mask for cns, duration: %.3f" % (t3 - t2))















