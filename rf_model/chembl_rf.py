# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Nov 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description:

import os
import sys
import math
import time
import getpass
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

sys.path.append("/home/%s/Documents/chembl/data_files/" % getpass.getuser())
import chembl_input as ci





# the newly picked out 15 targets, include 9 targets from 5 big group, and 6 targets from others.
target_list = ["CHEMBL279", "CHEMBL203", # Protein Kinases
               "CHEMBL217", "CHEMBL253", # GPCRs (Family A)
               "CHEMBL235", "CHEMBL206", # Nuclear Hormone Receptors
               "CHEMBL240", "CHEMBL4296", # Voltage Gated Ion Channels
               "CHEMBL4805", # Ligand Gated Ion Channels
               "CHEMBL204", "CHEMBL244", "CHEMBL4822", "CHEMBL340", "CHEMBL205", "CHEMBL4005" # Others
              ] 

# the target 
target = "CHEMBL203"


# 
model_dir = "model_files"
if not os.path.exists(model_dir):
  os.mkdir(model_dir)

#
pred_dir = "pred_files"
if not os.path.exists(pred_dir):
  os.mkdir(pred_dir)


def train_pred(target, train_pos_multiply=0):
  # 
  d = ci.Dataset(target, train_pos_multiply=train_pos_multiply)
  # random forest clf
  clf = RandomForestClassifier(n_estimators=100, max_features=1.0/3, n_jobs=10, max_depth=None, min_samples_split=5, random_state=0)
  # fit model
  clf.fit(d.train_features, d.train_labels)
  # save model
  joblib.dump(clf, model_dir + "/rf_%s.m" % target)
  # predict class probabilities
  #train_pred_proba = clf.predict_proba(d.train_features)[:, 1]
  test_pred_proba = clf.predict_proba(d.test_features)[:, 1]
  # save pred
  test_pred_file = open(pred_dir + "/test_%s.pred" % target, "w")
  for id_, pred_v, l_v in zip(d.time_split_test["CMPD_CHEMBLID"], test_pred_proba, d.test_labels):
    test_pred_file.write("%s\t%f\t%f\n" % (id_, pred_v, l_v))
  test_pred_file.close()
  # draw roc fig
  fpr, tpr, _ = roc_curve(d.test_labels, test_pred_proba)
  roc_auc = auc(fpr, tpr)
  plt.figure()
  plt.plot(fpr, tpr, color="r", lw=2, label="ROC curve (area = %.2f)" % roc_auc)
  plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver operating characteristic of RF model on %s" % target)
  plt.legend(loc="lower right")
  plt.savefig("%s.png" % target)
  #plt.show()





target_list = ["CHEMBL203", "CHEMBL205", "CHEMBL279", "CHEMBL340", 
               "CHEMBL4005", "CHEMBL4805",  
              ] 


for target, tpm in zip(target_list, tpm_list):
  t0 = time.time()
  train_pred(target, train_pos_multiply=0)
  t1 = time.time()
  print("%s duration: %.3f" % (target, t1-t0))


"""

t0 = time.time()
train_pred("CHEMBL4805", train_pos_multiply=0)
t1 = time.time()
print("%s duration: %.3f" % (target, t1-t0))

"""











"""
pns_pred = clf.predict(d.target_pns_features)
cns_pred = clf.predict(d.target_cns_features_train)
train_pred = clf.predict(d.train_features)
test_pred = clf.predict(d.test_features)

pns_result = ci.compute_performance(d.target_pns_mask.values.astype(int), pns_pred)
cns_result = ci.compute_performance(d.target_cns_mask_train.values.astype(int), cns_pred)
train_result = ci.compute_performance(d.train_labels, train_pred)
test_result = ci.compute_performance(d.test_labels, test_pred)

print(train_result)

print(test_result)
"""

# load model
#clf = joblib.load(model_dir + "/rf_%s.m" % target)




