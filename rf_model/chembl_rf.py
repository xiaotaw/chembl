# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Nov 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description:

import math
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


fp_files = "../data_files/fp_files"
mask_files = "../data_files/mask_files"
structure_files = "../data_files/structure_files"

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
f = open(fp_files + "/chembl.apfp", "r")
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
f = open(fp_files + "/pubchem_neg_sample.apfp", "r")
for line in f:
  id_, fps_str = line.split("\t")
  id_ = id_.strip()
  fps_str = fps_str.strip()
  pns_id.append(id_)
  pns_apfp[id_] = fps_str

f.close()



# read top 40 targets' label
#clf_label_40 = np.genfromtxt(structure_files + "/chembl_top40.label", usecols=[0, 2, 3, 4], delimiter="\t", skip_header=1, dtype=str)
clf_label_40 = pd.read_csv(structure_files + "/chembl_top40.label", usecols=[0, 2, 3, 4], delimiter="\t")

# the target 
target = target_list[0]
target_clf_label = clf_label_40[clf_label_40["TARGET_CHEMBLID"] == target]


# read mask
target_pns_mask = pd.Series.from_csv(mask_files + "/%s_pns.mask" % target, header=None, sep="\t")
target_cns_mask = pd.Series.from_csv(mask_files + "/%s_cns.mask" % target, header=None, sep="\t")
    
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


# generate features
target_pns_features = sparse_features([pns_apfp[k] for k in pns_id])[:, :-1]
target_chembl_features = sparse_features([chembl_apfp[k] for k in chembl_id])[:, :-1]

# time split
time_split_test = target_clf_label[target_clf_label["YEAR"]>2014]
m = target_cns_mask.index.isin(time_split_test["CMPD_CHEMBLID"])
target_chembl_features_test = target_chembl_features[m]
target_chembl_features_train = target_chembl_features[~m]
target_cns_mask_test = target_cns_mask[m]
target_cns_mask_train = target_cns_mask[~m]

# random forest clf
clf = RandomForestClassifier(n_estimators=100, max_features=1.0/3, n_jobs=10, max_depth=None, min_samples_split=5, random_state=0)


train_features = sparse.vstack([target_chembl_features_train, target_pns_features])
train_labels = np.hstack([target_cns_mask_train, target_pns_mask]).astype(int)

clf.fit(train_features, train_labels)



def compute_performance(label, prediction):
  """sensitivity(SEN), specificity(SPE), accuracy(ACC), matthews correlation coefficient(MCC) 
  """
  assert label.shape[0] == prediction.shape[0], "label number should be equal to prediction number"
  N = label.shape[0]
  APP = sum(prediction)
  ATP = sum(label)
  TP = sum(prediction * label)
  FP = APP - TP
  FN = ATP - TP
  TN = N - TP - FP - FN
  SEN = float(TP) / (ATP)
  SPE = float(TN) / (N - ATP)
  ACC = float(TP + TN) / N
  MCC = (TP * TN - FP * FN) / (math.sqrt(long(N - APP) * long(N - ATP) * APP * ATP)) if not (N - APP) * (N - ATP) * APP * ATP == 0 else 0.0
  return TP, TN, FP, FN, SEN, SPE, ACC, MCC

pns_pred = clf.predict(target_pns_features)
cns_pred = clf.predict(target_chembl_features_train)
train_pred = clf.predict(train_features)
test_pred = clf.predict(target_chembl_features_test)

pns_result = compute_performance(target_pns_mask.values.astype(int), pns_pred)
cns_result = compute_performance(target_cns_mask_train.values.astype(int), cns_pred)
train_result = compute_performance(train_labels, train_pred)
test_result = compute_performance(target_cns_mask_test.values.astype(int), test_pred)


pns_pred = clf.predict(target_pns_features)
pns_result = compute_performance(target_pns_mask, pns_pred)



# save model
joblib.dump(clf, "rf_%s.m" % target)

# load model
#clf = joblib.load("rf_%s.m" % target)




