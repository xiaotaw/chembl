#!/usr/bin/python
# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: calculate morgan fingerprint for target using rdkit,
#              inputs: smiles file
#              outputs: mgfp files which contain chembl_id, label, mgfp, year.

import os
import time
from rdkit import Chem
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprint, ExplainPairScore
from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprint, ExplainPathScore
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint

from rdkit.Chem import AllChem
#AllChem.Compute2DCoords(m2)

# for draw molecules
from rdkit.Chem import Draw
#Draw.MolToFile(mol, "mol_name.png")


if __name__ == "__main__":


  # 8 targets' pos sample

  structure_dir = "structure_files/"
  fp_dir = "fp_files/"
  if not os.path.exists(fp_dir):
    os.mkdir(fp_dir)
  # the maximum radius for morgan fingerprint
  radius = 6

  # max path length of Topological Torsion fingerprint
  max_path_len = 6

  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  for target in target_list:
    start = time.time()
    # read molecules from smiles file, and add H for each molecules.
    gz_supplier = Chem.SmilesMolSupplier(structure_dir + target+".smiles", delimiter="\t", titleLine=True)
    molecules = [Chem.AddHs(x) for x in gz_supplier if x is not None]

    # open file for fps
    mgfp_file = open(fp_dir + target+".mgfp6", "w")
    apfp_file = open(fp_dir + target+".apfp", "w")
    ttfp_file = open(fp_dir + target+".ttfp", "w")

    for mol in molecules:
      # read chembl_id, year, label
      cid = mol.GetProp("CHEMBL_ID") if mol.HasProp("CHEMBL_ID") else None
      year = mol.GetProp("YEAR") if mol.HasProp("YEAR") else None
      label = mol.GetProp("LABEL") if mol.HasProp("LABEL") else None
      # calculate morgan fingerprint, and save into file
      mgfps = GetMorganFingerprint(mol, radius).GetNonzeroElements()
      mgfp_file.write("%s\t%s\t%s\t%s\n" % (cid, str(mgfps), label, year))
      # calculate atom pair fingerprint, and save into file
      apfps = GetAtomPairFingerprint(mol).GetNonzeroElements()
      apfp_file.write("%s\t%s\n" % (cid, str(apfps)))
      # calculate Topological Torsion fingerprint, and save into file
      ttfps = {} 
      for i in range(1, max_path_len):
        ttfps.update(GetTopologicalTorsionFingerprint(mol, i).GetNonzeroElements())
      ttfp_file.write("%s\t%s\n" % (cid, str(ttfps)))

    # close fp files
    mgfp_file.close()
    apfp_file.close()
    ttfp_file.close()
    print("%s\t%5d\t%6.2f\n" % (target, len(molecules), time.time() - start))


  # pubchem neg sample  

  structure_dir = "structure_files/"
  fp_dir = "fp_files/"
  if not os.path.exists(fp_dir):
    os.mkdir(fp_dir)
  # the maximum radius for morgan fingerprint
  radius = 6

  start = time.time()

  target = "pubchem_neg_sample"
  sup = Chem.SDMolSupplier(structure_dir + "pubchem_neg_sample.sdf")
  molecules = [Chem.AddHs(x) for x in sup if x is not None] 

  # open file for fps
  mgfp_file = open(fp_dir + target+".mgfp6", "w")
  apfp_file = open(fp_dir + target+".apfp", "w")
  ttfp_file = open(fp_dir + target+".ttfp", "w")

  for mol in molecules:
    # read chembl_id, year, label
    cid = mol.GetProp("PUBCHEM_COMPOUND_CID") if mol.HasProp("PUBCHEM_COMPOUND_CID") else None
    # calculate morgan fingerprint, and save into file
    mgfps = GetMorganFingerprint(mol, radius).GetNonzeroElements()
    mgfp_file.write("%s\t%s\n" % (cid, str(mgfps)))
    # calculate atom pair fingerprint, and save into file
    apfps = GetAtomPairFingerprint(mol).GetNonzeroElements()
    apfp_file.write("%s\t%s\n" % (cid, str(apfps)))
    # calculate Topological Torsion fingerprint, and save into file
    ttfps = {} 
    for i in range(1, max_path_len):
      try:
        ttfps.update(GetTopologicalTorsionFingerprint(mol, i).GetNonzeroElements())
      except IndexError:
        print("IndexError accur on a molecule with i=%d" % i)
    ttfp_file.write("%s\t%s\n" % (cid, str(ttfps)))
  # close fp files
  mgfp_file.close()
  apfp_file.close()
  ttfp_file.close()
  print("%s\t%5d\t%6.2f\n" % (target, len(molecules), time.time() - start))


  """
  # pubchem all
  structure_dir = "structure_files/"
  fp_dir = "fp_files/"
  if not os.path.exists(fp_dir):
    os.mkdir(fp_dir)
  # the maximum radius for morgan fingerprint
  radius = 6
  """



