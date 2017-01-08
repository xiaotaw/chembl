# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Nov 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description: 1. calculate atom pair fingerprint(apfp) for chembl molecules
#              2. analyse apfp

import os
import numpy as np
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprint

def dict_2_str(d):
  keylist = d.keys()
  keylist.sort()
  kv_list = ["{}: {}".format(k, d[k]) for k in keylist] 
  return ", ".join(kv_list)

"""
def str_2_dict(fps_str):
  fps_dict = {}
  for fp in fps_str[1:-1].split(","):
    if ":" in fp:
      k, v = fp.split(":")
      fps_dict[int(k)] = int(v)
  return fps_dict
"""

## calculate chembl apfp
#
sup = Chem.SmilesMolSupplier("structure_files/chembl.smiles", delimiter=",", smilesColumn=1, nameColumn=0, titleLine=True)

if not os.path.exists("fp_files"):
  os.mkdir("fp_files")

apfp_file = open("fp_files/chembl.apfp", "w")

for m in sup:
  if m is None:
    continue
  id_ = m.GetProp("_Name")
  apfps = GetAtomPairFingerprint(Chem.RemoveHs(m)).GetNonzeroElements()
  apfp_file.write("%s\t{%s}\n" % (id_, dict_2_str(apfps)))

apfp_file.close()


## calculate pns apfp
#
sup = Chem.SDMolSupplier("structure_files/pubchem_neg_sample.sdf")

apfp_file = open("fp_files/pubchem_neg_sample.apfp", "w")

for m in sup:
  if m is None:
    continue
  id_ = m.GetProp("PUBCHEM_COMPOUND_CID")
  apfps = GetAtomPairFingerprint(Chem.RemoveHs(m)).GetNonzeroElements()
  apfp_file.write("%s\t{%s}\n" % (id_, dict_2_str(apfps)))

apfp_file.close()


##pick out some apfps
#Descrition:
#  After calculate fps, I found that many fps was so rare that only few 
#  molecule has those fps, and those fps account the major part.
#  So, I sum up all apfps' frequency, and remove fps with frequence <= 10,
#  or frequence > n-10(n is the number of sample).
#Output:
#  apfp.picked_all: contains all apfps picked out for 8 target and pubchem neg sample(pns).
"""
# for pns 
f = open("fp_files/pubchem_neg_sample.apfp", "r")
counts = defaultdict(lambda : 0)
n = 0
for line in f:
  fps = line.split("\t")[1].strip()[1:-1].split(",")
  n += 1
  for fp in fps:
    if ":" in fp:
      k, v = fp.split(":")
      counts[int(k)] += 1
f.close()
v = [[k, counts[k]] for k in counts.keys()]
v = np.array(v)
m = (v[:, 1] > 10) & (v[:, 1] < n - 10)
pns_k = v[:, 0][m]

# output into file
f = open("fp_files/pns_apfp.picked_all", "w")
for i in k:
  f.write("%d\n" % i)
f.close()


# for chembl
f = open("fp_files/chembl.apfp", "r")
counts = defaultdict(lambda : 0)
n = 0
for line in f:
  fps = line.split("\t")[1].strip()[1:-1].split(",")
  n += 1
  if n % 10000 == 0:
    print(("line num: %d, unique fps num: %d") % (n, len(counts.keys())))
  for fp in fps:
    if ":" in fp:
      k, v = fp.split(":")
      counts[int(k)] += 1

f.close()
print(("line num: %d, unique fps num: %d") % (n, len(counts.keys())))

v = [[k, counts[k]] for k in counts.keys()]
v = np.array(v)
m = (v[:, 1] > 10) & (v[:, 1] < n - 10)
pns_k = v[:, 0][m]

# output into file
f = open("fp_files/chembl_apfp.picked_all", "w")
for i in k:
  f.write("%d\n" % i)
f.close()

"""









