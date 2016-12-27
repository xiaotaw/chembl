# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Nov 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description: 1. clean chembl data,
#              2. all molecules' smiles,
#              3. all bioactivity records' label for classification(clf label).

import os
import numpy as np
import pandas as pd


txt_dir = "txt_files"
structure_dir = "structure_files"
if not os.path.exists(structure_dir):
  os.mkdir(structure_dir)


# read all chembl bioactivity records
chembl = pd.read_csv(os.path.join(txt_dir, "chembl_bioactivity_all.txt"), delimiter="\t")


# about smiles
#
smiles_all = chembl[["CMPD_CHEMBLID", "CANONICAL_SMILES"]]
ss = smiles_all.astype(str)
ss_nan = ss[ss["CANONICAL_SMILES"] == "nan"]
ss_str = ss[~(ss["CANONICAL_SMILES"] == "nan")]
# it is equal to 0, meaning that the molecules missing "CANONICAL_SMILES".
# so return to the beginning, I will remove the molecules without "CANONICAL_SMILES".
assert ss_str["CMPD_CHEMBLID"].isin(ss_nan["CMPD_CHEMBLID"]).sum() == 0
# remove molecules without "CANONICAL_SMILES"
m = chembl["CANONICAL_SMILES"].astype(str) != "nan"
chembl = chembl[m]


# there are still 400 duplicates, however it doesn't matter.
#dup = ss_str[ss_str.duplicated(subset=["CANONICAL_SMILES"], keep=False)]
#dup = dup.sort_values(by=["CANONICAL_SMILES"])
smiles = chembl.drop_duplicates(subset=["CMPD_CHEMBLID", "CANONICAL_SMILES"])[["CMPD_CHEMBLID", "CANONICAL_SMILES"]]
# sort smile by ID
smiles["id_num"] = smiles["CMPD_CHEMBLID"].map(lambda x: int(x[6:])) 
smiles.sort_values(by=["id_num"], inplace=True)
smiles.reset_index(drop=True, inplace=True)
smiles = smiles[["CMPD_CHEMBLID", "CANONICAL_SMILES"]]
# save into files
smiles.to_csv(os.path.join(structure_dir, "chembl.smiles"), index=False)


# about classification(clf) labels
#
inhibitor = chembl[chembl["STANDARD_TYPE"].isin(["IC50", "Ki", "EC50"])]
# transfer to standard units and standard values, records with invalid value are removed
inhibitor = inhibitor[np.isfinite(inhibitor["STANDARD_VALUE"].astype(np.float32))]
m = inhibitor["STANDARD_UNITS"].isin(["/uM"])
inhibitor[m]["STANDARD_VALUE"] *= 1000
inhibitor[m]["STANDARD_UNITS"] = "nM"
m = inhibitor["STANDARD_UNITS"].isin(["/nM", "ug nM-1", "Ke nM-1"])
inhibitor[m]["STANDARD_TYPE"] = "nM"
# NOTE: some molecules' "MOLWEIGHT" is unknown(np.nan), 
# and their "STANDARD_VALUE" will be np.nan
m = inhibitor["STANDARD_UNITS"].isin(["ug.mL-1"])
inhibitor[m]["STANDARD_VALUE"] = inhibitor[m]["STANDARD_VALUE"] / inhibitor[m]["MOLWEIGHT"] * 10**6
inhibitor[m]["STANDARD_UNITS"] = "nM"

inhibitor = inhibitor[inhibitor["STANDARD_UNITS"].isin(["nM"]) & np.isfinite(inhibitor["STANDARD_VALUE"].astype(np.float32))]


# judge a record's clf label
def is_pos(row):
  r = row["RELATION"]
  v = np.float32(row["STANDARD_VALUE"])
  if r == "<" or r == "<=":
    return 1 if v <= 10000 else 0
  elif r == ">" or r == ">=":
    return -1 if v >= 10000 else 0
  elif r == "=":
    return 1 if v <= 10000 else -1
  else:
    return np.nan

inhibitor["CLF_LABEL"] = inhibitor.apply(is_pos, axis=1)
inhibitor = inhibitor[np.isfinite(inhibitor["CLF_LABEL"])]

# group
grouped = inhibitor.groupby(by=["TARGET_CHEMBLID", "PREF_NAME", "CMPD_CHEMBLID"], as_index=False)
# judge one molecule's label by the average label
clf_label_all = grouped[["CLF_LABEL", "YEAR"]].mean()
# inconclusive reocrds are removed
clf_label = clf_label_all[(clf_label_all["CLF_LABEL"] > 0.5) | (clf_label_all["CLF_LABEL"] < -0.5)]
# save into file
clf_label.to_csv(os.path.join(structure_dir, "chembl.label"), index=False, sep="\t")
# sort target by number of records
lss = clf_label.groupby(by=["TARGET_CHEMBLID", "PREF_NAME"], as_index=False).size()

#lss_1 = clf_label.groupby(by=["TARGET_CHEMBLID"], as_index=False).size()

lss.sort_values(ascending=False, inplace=True)


target_list = lss[lss >= 3000].index
clf_label_used = clf_label[clf_label["TARGET_CHEMBLID"].isin(target_list)]
clf_label_used.to_csv(os.path.join(structure_dir, "chembl_top40.label"), sep="\t", index=False)

    
target_list = lss[lss >= 2000].index
clf_label_used = clf_label[clf_label["TARGET_CHEMBLID"].isin(target_list)]
clf_label_used.to_csv(os.path.join(structure_dir, "chembl_top79.label"), sep="\t", index=False)




