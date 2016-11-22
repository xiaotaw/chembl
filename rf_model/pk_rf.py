# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Oct 2016
# Time Last Updated: Oct 2016
# Addr: Shenzhen, China
# Description: using random forest for pk

import math
import time
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import pk_input as pki


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
  MCC = (TP * TN - FP * FN) / (math.sqrt((N - APP) * (N - ATP) * APP * ATP)) if not (N - APP) * (N - ATP) * APP * ATP == 0 else 0.0
  return TP, TN, FP, FN, SEN, SPE, ACC, MCC


if __name__ == "__main__":

  # title 
  title_str = " type     target     TP     FN     TN     FP     SEN     SPE     ACC     MCC  duration"
  print(title_str)

  # log file
  log_file = open("pk_rf_100.log", "w")
  log_file.write(title_str + "\n")

  # 8 protein kinase target's abbrivate names
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  # create input dataset
  d = pki.Datasets(target_list, one_hot=False)


  for target in target_list:
    t0 = time.time()
    # train features(x) and labels(y)
    train_x = np.vstack([d.pos[target].features[d.pos[target].train_perm], d.neg.features[d.neg.train_perm]])
    train_y = np.hstack([d.pos[target].labels[d.pos[target].train_perm], d.neg.mask_dict[target][d.neg.train_perm]])

    # test features(x) and labels(y)
    test_x = np.vstack([d.pos[target].features[d.pos[target].test_perm], d.neg.features[d.neg.test_perm]])
    test_y = np.hstack([d.pos[target].labels[d.pos[target].test_perm], d.neg.mask_dict[target][d.neg.test_perm]])

    # random forest classifier
    clf = RandomForestClassifier(n_estimators=100, max_features=1.0/3, n_jobs=8, max_depth=None, min_samples_split=5, random_state=0)

    # fit the model
    clf.fit(train_x, train_y)

    # evaluate the model on train data
    train_pred = clf.predict(train_x)
    TP, TN, FP, FN, SEN, SPE, ACC, MCC = compute_performance(train_y, train_pred)
    t1 = time.time()

    format_str = "%5s %10s %6d %6d %6d %6d %6.5f %6.5f %6.5f %6.5f %5f"
    print(format_str % ("train", target, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0))
    log_file.write(format_str % ("train", target, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0))
    log_file.write("\n")

    # evaluate the model on test data
    test_pred = clf.predict(test_x)
    TP, TN, FP, FN, SEN, SPE, ACC, MCC  = compute_performance(test_y, test_pred)
    t2 = time.time()
    print(format_str % ("test", target, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t2-t0))
    log_file.write(format_str % ("test", target, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t2-t0))
    log_file.write("\n")

  log_file.close()

