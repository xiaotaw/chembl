# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Nov 2016
# Time Last Updated: Nov 2016
# Addr: Shenzhen, China
# Description:

import numpy as np
import pandas as pd


def str_2_dict(fps_str):
  fps_dict = {}
  for fp in fps_str[1:-1].split(","):
    if ":" in fp:
      k, v = fp.split(":")
      fps_dict[int(k)] = int(v)
  return fps_dict


#target_list = ['CHEMBL203', 'CHEMBL204', 'CHEMBL205', 'CHEMBL214', 'CHEMBL217',
#       'CHEMBL218', 'CHEMBL220', 'CHEMBL224', 'CHEMBL225', 'CHEMBL226',
#       'CHEMBL228', 'CHEMBL230', 'CHEMBL233', 'CHEMBL234', 'CHEMBL235',
#       'CHEMBL236', 'CHEMBL237', 'CHEMBL240', 'CHEMBL244', 'CHEMBL251',
#       'CHEMBL253', 'CHEMBL256', 'CHEMBL259', 'CHEMBL260', 'CHEMBL261',
#       'CHEMBL264', 'CHEMBL267', 'CHEMBL279', 'CHEMBL284', 'CHEMBL2842',
#       'CHEMBL289', 'CHEMBL325', 'CHEMBL332', 'CHEMBL333', 'CHEMBL340',
#       'CHEMBL344', 'CHEMBL4005', 'CHEMBL4296', 'CHEMBL4722', 'CHEMBL4822']

class Dataset(object):
  def __init__(self, target):
    # read top 40 compound number targets' clf label
    clf_label_all = pd.read_csv("structure_files/chembl_top40.label", delimiter="\t")

    target_list = clf_label_all["TARGET_CHEMBLID"].unique()

    # fps
    fps_all = pd.Series.from_csv("fp_files/chembl.apfp", header=None, sep="\t")
    #apfp_picked_all = np.genfromtxt("fp_files/apfp.picked_all", dtype=int)

    # pubchem neg sample
    pns = pd.Series.from_csv("fp_files/pubchem_neg_sample.apfp", header=None, sep="\t")
    pns_features = pd.DataFrame(index=pns.index, data=[str_2_dict(x) for x in pns], dtype=int)
    pns_counts = pns_features.count()




    target = target_list[0]
    clf_label = clf_label_all[clf_label_all["TARGET_CHEMBLID"] == target]
    fps = fps_all.ix[clf_label["CMPD_CHEMBLID"].unique()] 
    target_features = pd.DataFrame(index=fps.index, data=[str_2_dict(x) for x in fps], dtype=int)

    target_counts = target_features.count()
    counts = pns_counts.add(target_counts, fill_value=0)

    apfp_picked = counts[counts > 10].index







