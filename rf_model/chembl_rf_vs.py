# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Dec 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description:

import os
import sys
import getpass
import numpy as np
from scipy import sparse
from collections import defaultdict
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

# the target 
target = "CHEMBL205"

# input dataset
d = ci.DatasetVS(target)

# read saved rf clf model
clf = joblib.load("model_files/rf_%s.m" % target)

# pred file
pred_dir = "pred_files/%s" % target

for part_num in range(13):

  pred_path = os.path.join(pred_dir, "vs_pubchem_%d.pred" % part_num)
  predfile = open(pred_path, "w")

  fp_dir = "/raid/xiaotaw/pubchem/fp_files/%d" % num
  for i in range(num * 10000000 + 1, (num + 1) * 10000000, 25000):
    fn = os.path.join(fp_dir, "Compound_{:0>9}_{:0>9}.apfp".format(i, i + 24999))
    if os.path.exists(fn):
      d.reset(fp_fn)
      features = d.features_dense
      pred = clf.predict_prob(features)
      for id_, pred_v in zip(d.pubchem_id, pred[:, 1]):
        predfile.writeline("%s\t%f\n" % (id_, pred_v))
      print("%s\t%d\n" % (fn, pred.shape[0]))





