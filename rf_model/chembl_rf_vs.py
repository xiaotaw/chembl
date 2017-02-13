# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Dec 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description:

import os
import sys
import time
import getpass
import numpy as np
from scipy import sparse
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.externals import joblib
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


target_list = ["CHEMBL203", "CHEMBL204", "CHEMBL205", "CHEMBL244", "CHEMBL279", "CHEMBL340", 
                 "CHEMBL4005", "CHEMBL4805", "CHEMBL4822", 
                ] 

def virtual_screening(target):
  # input dataset
  d = ci.DatasetVS(target)
  # read saved rf clf model
  clf = joblib.load("model_files/rf_%s.m" % target)
  # pred file
  pred_dir = "pred_files/%s" % target
  if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
  for part_num in range(13):
    t0 = time.time()
    pred_path = os.path.join(pred_dir, "vs_pubchem_%d.pred" % part_num)
    predfile = open(pred_path, "w")
    fp_dir = "/raid/xiaotaw/pubchem/fp_files/%d" % part_num
    for i in range(part_num * 10000000 + 1, (part_num + 1) * 10000000, 25000):
      fp_fn = os.path.join(fp_dir, "Compound_{:0>9}_{:0>9}.apfp".format(i, i + 24999))
      if os.path.exists(fp_fn):
        d.reset(fp_fn)
        features = d.features_dense
        pred = clf.predict_proba(features)
        for id_, pred_v in zip(d.pubchem_id, pred[:, 1]):
          predfile.write("%s\t%f\n" % (id_, pred_v))
        #print("%s\t%d\n" % (fp_fn, pred.shape[0]))
    t1 = time.time()
    print("%s %d: %.3f" %(target, part_num, t1-t0))


def analyse(target):
  vs_pred_file = "pred_files/%s/vs_pubchem.pred" % (target)
  if not os.path.exists(vs_pred_file):
    os.system("cat pred_files/%s/vs_pubchem_*.pred > pred_files/%s/vs_pubchem.pred" % (target, target))
  aa = np.genfromtxt(vs_pred_file, delimiter="\t")
  a = aa[:, 1]
  test_pred_file = "pred_files/test_%s.pred" % (target)
  bb = np.genfromtxt(test_pred_file, delimiter="\t", usecols=[1,2])
  b = bb[:, 0][bb[:, 1].astype(bool)]
  x = []
  y = []
  for i in range(10):
    mark = (i + 1) / 20.0
    xi = 1.0 * (b > mark).sum() / b.shape[0]
    yi = (a > mark).sum()
    x.append(xi)
    y.append(yi)
  plt.plot(x, y, "*")
  plt.xlabel("pos yeild rate")
  plt.ylabel("vs pubchem false pos")
  plt.savefig("pred_files/%s/analyse.png" % (target))


target = target_list[int(sys.argv[1])]
virtual_screening(target)
analyse(target)


"""
for target in target_list:
  virtual_screening(target)
  #analyse(target)
"""
