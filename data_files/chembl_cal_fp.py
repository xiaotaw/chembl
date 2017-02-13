# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Nov 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description: 1. calculate atom pair fingerprint(apfp) for chembl molecules
#              2. analyse apfp

import os
import gzip
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
"""

## calculate ChemDiv apfp
ChemDiv_dir = "/raid/xiaotaw/ChemDiv"
fn_list = ["DC01_350000.sdf", "DC02_350000.sdf", "DC03_222773.sdf", "DC_saltdata_not-available_124145.sdf", "IC_non-excl_82693.sdf", "NC_340320.sdf"]

for fn in fn_list:
  gzsup = Chem.SDMolSupplier(ChemDiv_dir + "/" + fn)
  molecules = [x for x in gzsup if x is not None]
  apfp_file = open(ChemDiv_dir + "/" + fn.replace("sdf", "apfp"), "w")
  for mol in molecules:
    id_ = mol.GetProp("IDNUMBER")
    apfps = GetAtomPairFingerprint(Chem.RemoveHs(mol)).GetNonzeroElements()
    apfp_file.write("%s\t{%s}\n" % (id_, dict_2_str(apfps)))
  apfp_file.close()






