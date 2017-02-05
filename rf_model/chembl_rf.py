# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Nov 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description:

import sys
import math
import getpass
import numpy as np
import pandas as pd
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

d = ci.Dataset(target, train_pos_multiply=0)


# random forest clf
clf = RandomForestClassifier(n_estimators=100, max_features=1.0/3, n_jobs=10, max_depth=None, min_samples_split=5, random_state=0)

# fit model
clf.fit(d.train_features, d.train_labels)

# save model
joblib.dump(clf, "rf_%s.m" % target)

# predict class probabilities
#pns_pred_proba = clf.predict_proba(d.target_pns_features)[:, 1]
#cns_pred_proba = clf.predict_proba(d.target_cns_features_train)[:, 1]
train_pred_proba = clf.predict_proba(d.train_features)[:, 1]
test_pred_proba = clf.predict_proba(d.test_features)[:, 1]


#pns_pred = clf.predict(d.target_pns_features)
#cns_pred = clf.predict(d.target_cns_features_train)
#train_pred = clf.predict(d.train_features)
#test_pred = clf.predict(d.test_features)

#pns_result = ci.compute_performance(d.target_pns_mask.values.astype(int), pns_pred)
#cns_result = ci.compute_performance(d.target_cns_mask_train.values.astype(int), cns_pred)
#train_result = ci.compute_performance(d.train_labels, train_pred)
#test_result = ci.compute_performance(d.test_labels, test_pred)



# load model
#clf = joblib.load("rf_%s.m" % target)




