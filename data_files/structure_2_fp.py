# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Nov 2016
# Addr: Shenzhen, China
# Description: calculate morgan fingerprint for target using rdkit,
#              inputs: smiles or sdf file
#              outputs: mgfp, apfp, ttfp files which contain chembl_id, label, mgfp, year.

import os
import time
import numpy as np
#import matplotlib.pyplot as plt
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Draw # for draw molecules
from rdkit.Chem import AllChem # for AllChem.Compute2DCoords()
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprint, ExplainPairScore
from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprint, ExplainPathScore


def cal_pos():
  """caculate fps for 8 targets' pos sample
  """
  structure_dir = "structure_files/"
  fp_dir = "fp_files/"
  if not os.path.exists(fp_dir):
    os.mkdir(fp_dir)
  # the maximum radius for morgan fingerprint
  radius = 6
  # max path length of Topological Torsion fingerprint
  min_path_len = 2
  max_path_len = 6
  # 8 targets
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]
  for target in target_list:
    start = time.time()
    # read molecules from smiles file, and remove H for each molecules.
    gz_supplier = Chem.SmilesMolSupplier(structure_dir + target+".smiles", delimiter="\t", titleLine=True)
    molecules = [Chem.RemoveHs(x) for x in gz_supplier if x is not None]
    # open file for fps
    response_file = open(fp_dir + target + ".response", "w")
    mgfp_file = open(fp_dir + target + ".mgfp6", "w")
    apfp_file = open(fp_dir + target + ".apfp", "w")
    ttfp_file = open(fp_dir + target + ".ttfp", "w")
    fmgfp_file = open(fp_dir + target + ".fmgfp6", "w")
    for mol in molecules:
      # read chembl_id, year, label
      cid = mol.GetProp("CHEMBL_ID") if mol.HasProp("CHEMBL_ID") else None
      year = mol.GetProp("YEAR") if mol.HasProp("YEAR") else None
      label = mol.GetProp("LABEL") if mol.HasProp("LABEL") else None
      type_ = mol.GetProp("TYPE") if mol.HasProp("TYPE") else None
      relation = mol.GetProp("RELATION") if mol.HasProp("RELATION") else None
      value = mol.GetProp("VALUE") if mol.HasProp("VALUE") else None
      response_file.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (cid, year, label, type_, relation, value))
      # calculate morgan fingerprint, and save into file
      fmgfps = GetMorganFingerprint(mol, radius, useFeatures=True).GetNonzeroElements()
      fmgfp_file.write("%s\t%s\n" % (cid, str(fmgfps)))
      # calculate morgan fingerprint, and save into file
      mgfps = GetMorganFingerprint(mol, radius).GetNonzeroElements()
      mgfp_file.write("%s\t%s\n" % (cid, str(mgfps)))
      # calculate atom pair fingerprint, and save into file
      apfps = GetAtomPairFingerprint(mol).GetNonzeroElements()
      apfp_file.write("%s\t%s\n" % (cid, str(apfps)))
      # calculate Topological Torsion fingerprint, and save into file
      ttfps = {} 
      for i in range(min_path_len, max_path_len):
        ttfps.update(GetTopologicalTorsionFingerprint(mol, i).GetNonzeroElements())
      ttfp_file.write("%s\t%s\n" % (cid, str(ttfps)))
    # close fp files
    response_file.close()
    mgfp_file.close()
    apfp_file.close()
    ttfp_file.close()
    fmgfp_file.close()
    print("%s\t%5d\t%6.2f\n" % (target, len(molecules), time.time() - start))
    

def cal_neg():
  """caculate fps for pubchem neg sample  
  """
  start = time.time()
  structure_dir = "structure_files/"
  fp_dir = "fp_files/"
  if not os.path.exists(fp_dir):
    os.mkdir(fp_dir)
  # the maximum radius for morgan fingerprint
  radius = 6
  # max path length of Topological Torsion fingerprint
  min_path_len = 2
  max_path_len = 6
  # read mol from sdf file
  target = "pubchem_neg_sample"
  sup = Chem.SDMolSupplier(structure_dir + "pubchem_neg_sample.sdf")
  molecules = [Chem.RemoveHs(x) for x in sup if x is not None] 
  # open file for fps
  mgfp_file = open(fp_dir + target+".mgfp6", "w")
  apfp_file = open(fp_dir + target+".apfp", "w")
  ttfp_file = open(fp_dir + target+".ttfp", "w")
  fmgfp_file = open(fp_dir + target+".fmgfp6", "w")
  for mol in molecules:
    # read chembl_id, year, label
    cid = mol.GetProp("PUBCHEM_COMPOUND_CID") if mol.HasProp("PUBCHEM_COMPOUND_CID") else None
    # calculate morgan fingerprint, and save into file
    fmgfps = GetMorganFingerprint(mol, radius, useFeatures=True).GetNonzeroElements()
    fmgfp_file.write("%s\t%s\n" % (cid, str(fmgfps)))
    # calculate morgan fingerprint, and save into file
    mgfps = GetMorganFingerprint(mol, radius).GetNonzeroElements()
    mgfp_file.write("%s\t%s\n" % (cid, str(mgfps)))
    # calculate atom pair fingerprint, and save into file
    apfps = GetAtomPairFingerprint(mol).GetNonzeroElements()
    apfp_file.write("%s\t%s\n" % (cid, str(apfps)))
    # calculate Topological Torsion fingerprint, and save into file
    ttfps = {} 
    for i in range(min_path_len, max_path_len):
      try:
        ttfps.update(GetTopologicalTorsionFingerprint(mol, i).GetNonzeroElements())
      except IndexError:
        print("IndexError accur on a molecule with i=%d" % i)
    ttfp_file.write("%s\t%s\n" % (cid, str(ttfps)))
  # close fp files
  mgfp_file.close()
  apfp_file.close()
  ttfp_file.close()
  fmgfp_file.close()
  print("%s\t%5d\t%6.2f\n" % (target, len(molecules), time.time() - start))


def pick_apfp():
  """pick out some apfps
  Descrition:
    After calculate fps, I found that many fps was so rare that only few 
    molecule has those fps, and those fps account the major part.
    So, I sum up all apfps' frequency, and remove fps with frequence <= 10,
    or frequence > n-10(n is the number of sample).
  Output:
    apfp.picked_all: contains all apfps picked out for 8 target and pubchem neg sample(pns).
  """
  # for pns 
  target = "pubchem_neg_sample"
  f = open("fp_files/" + target + ".apfp", "r")
  counts = defaultdict(lambda : 0)
  n = 0
  for line in f:
    fps = line.split("\t")[1][1:-1].split(",")
    n += 1
    for fp in fps:
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

  # for 8 target
  k_dict = {}
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]
  for target in target_list:
    print(target)
    counts = defaultdict(lambda : 0)
    f = open("fp_files/" + target + ".apfp", "r")
    n = 0
    for line in f:
      n += 1
      fps = line.split("\t")[1].split("{")[1].split("}")[0].split(",")
      for fp in fps:
        if fp == "":
          continue
        k, v = fp.split(":")
        counts[int(k)] += 1
    f.close()
    v = [[k, counts[k]] for k in counts.keys()]
    v = np.array(v)
    m = (v[:, 1] > 10) & (v[:, 1] < n - 10)
    k = v[:, 0][m]
    k_dict[target] = k
  
  for t in target_list:
    k = k_dict[t]
    f = open("fp_files/" + t + "_apfp.picked_all", "w")
    for i in k:
      f.write("%d\n" % i)
    f.close()
  
  # merge 8 targets' picked apfp, then merge with pns,
  # after that write into files
  m = set()
  for t in target_list:
    m = m | set(k_dict[t])

  all_k = set(m) | set(pns_k)
  f = open("fp_files/apfp.picked_all", "w")
  for i in all_k:
    f.write("%d\n" % i)
  f.close()
  

if __name__ == "__main__":
  cal_pos()
  cal_neg()
  # pick_apfp()








