# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Dec 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description:

import os
import numpy as np
from scipy import sparse
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier



mask_files = "../data_files/mask_files"

target_list = ['CHEMBL203', 'CHEMBL204', 'CHEMBL205', 'CHEMBL214', 'CHEMBL217',
               'CHEMBL218', 'CHEMBL220', 'CHEMBL224', 'CHEMBL225', 'CHEMBL226',
               'CHEMBL228', 'CHEMBL230', 'CHEMBL233', 'CHEMBL234', 'CHEMBL235',
               'CHEMBL236', 'CHEMBL237', 'CHEMBL240', 'CHEMBL244', 'CHEMBL251',
               'CHEMBL253', 'CHEMBL256', 'CHEMBL259', 'CHEMBL260', 'CHEMBL261',
               'CHEMBL264', 'CHEMBL267', 'CHEMBL279', 'CHEMBL284', 'CHEMBL2842',
               'CHEMBL289', 'CHEMBL325', 'CHEMBL332', 'CHEMBL333', 'CHEMBL340',
               'CHEMBL344', 'CHEMBL4005', 'CHEMBL4296', 'CHEMBL4722', 'CHEMBL4822']

# the target
target = target_list[0]

# read count and the apfps that were picked out 
counts = np.genfromtxt(mask_files + "/%s_apfp.count" % target, delimiter="\t", dtype=int)
target_apfp_picked = counts[counts[:, 1] > 10][:, 0]
target_apfp_picked.sort()

# columns and sparse features
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

# read saved rf clf model
clf = joblib.load("rf_%s.m" % target)

# open vs log
log_file = open("rf_%s_vs.log" % target, "w")

for num in range(13):
  fp_dir = "/raid/xiaotaw/pubchem/fp_files/%d" % num
  for i in range(num * 10000000 + 1, (num + 1) * 10000000, 25000):
    fn = os.path.join(fp_dir, "Compound_{:0>9}_{:0>9}.apfp".format(i, i + 24999))
    if os.path.exists(fn):
      ids_list = []
      fps_dict = {}
      f = open(fn, "r")
      for line in f:
        id_, fps_str = line.split("\t")
        id_ = id_.strip()
        fps_str = fps_str.strip()
        ids_list.append(id_)
        fps_dict[id_] = fps_str
      f.close()
      features = sparse_features([fps_dict[k] for k in ids_list])[:, :-1]
      pred = clf.predict(features)
      result = np.array(ids_list)[pred.astype(bool)]
      log_file.writelines(["%s\n" % x for x in result])
      print("%s\t%d\n" % (fn, result.shape[0]))


log_file.close()




