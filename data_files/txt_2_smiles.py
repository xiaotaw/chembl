# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Nov 2016
# Addr: Shenzhen, China
# Description: pick out pos and neg molecules for a target,
#              inputs: txt file downloaded from chembl
#              outputs: smiles file

import os

def is_pos(line):
  """
  # if <= 10 uM(10,000 nM), then pos, return 1,
  # elif > 100 uM(100,000 nM), then neg, return 0,
  # else then inconclussive, return -1,
  # if line[14] the standard value is missing or not numerical, return -2,
  # if line[15] the standart unit is not nM or ug.mL-1, return -3,
  # other error, return -4
  """
  mol_wt = line[6]
  std_type = line[12]
  relation = line[13]
  std_value = line[14]
  std_unit = line[15]
  # no value error
  if std_value == "" or std_value == None:
    return -2
  std_value = float(std_value)
  # using nM as std unit
  if std_unit == "ug.ml-1" or std_unit == "ug.mL-1":
    if mol_wt == "" or mol_wt == None:
      return -2
    std_value = std_value / float(mol_wt) * 1000 * 1000
  elif not std_unit == "nM":
    return -3
  # pick out pos mol, return with 1
  if relation == "<":
    return 1 if std_value <= 10000 else -1
  elif relation == "=":
    if std_value <= 10000:
      return 1
    elif std_value > 10000:
      return 0
    else:
      return -2
  elif relation == ">":
    return 0 if std_value >= 10000 else -1
  return -4


def txt2smile(in_file, out_file):
  """ select pos and neg compounds from txt file, which is downloaded from chembl.
  Args:
    in_file: <type 'str'> input txt file name, 
      and the file contains compounds only from one specific target,
      and all the compounds should has been filtered by IC50.
    out_file: <type 'str'> output smiles file name.

  Return: None
  """
  infile = open(in_file, "r")
  lines = infile.readlines()
  lines = [x.split("\t") for x in lines[1:]]
  infile.close()
  outfile = open(out_file, "w")

  outfile.write("SMILES\tCHEMBL_ID\tCHEMBL_ID\tYEAR\tLABEL\n")

  for line in lines:
    label = is_pos(line)
    if is_pos(line) != -1:
      outfile.write("%s\t%s\t%s\t%s\t%d\n" % (line[10], line[0], line[0], line[48], label))

  outfile.close()



def txt2smiles_1():
  """ an other version of txt2smile,
      No input and output parameters.
  """
  # define directory
  txt_dir = "txt_downloaded_from_chembl"
  structure_dir = "structure_files"
  if not os.path.exists(structure_dir):
    os.mkdir(structure_dir)

  # read inputs
  infile = open(os.path.join(txt_dir, "8_protein_kinase.txt"), "r")
  lines = infile.readlines()
  lines = [x.split("\t") for x in lines[1:]]
  infile.close()

  # define abbr for target name
  abbr_dict = {
    "Cyclin-dependent kinase 2": "cdk2",
    "Epidermal growth factor receptor erbB1": "egfr_erbB1",
    "Glycogen synthase kinase-3 beta": "gsk3b",
    "Hepatocyte growth factor receptor": "hgfr",
    "MAP kinase p38 alpha": "map_k_p38a",
    "Tyrosine-protein kinase LCK": "tpk_lck",
    "Tyrosine-protein kinase SRC": "tpk_src",
    "Vascular endothelial growth factor receptor 2": "vegfr2",
  }

  # open smiles file for each target
  outfile_dict = dict()
  for k in abbr_dict.keys():
    outfile_dict[k] = open(os.path.join(structure_dir, abbr_dict[k]+".smiles"), "w")
    outfile_dict[k].write("SMILES\tCHEMBL_ID\tCHEMBL_ID\tYEAR\tLABEL\tTYPE\tRELATION\tVALUE\n")
 
  # read info for each molecule(line)
  #for line in lines:
  for i in range(len(lines)):
    line = lines[i]
    if line[12] == "IC50" or line[12] == "ic50":
      label = is_pos(line)
      if label >= -1:
        # outfile write like: smiles, chemblid, chemblid, year, label
        outfile_dict[line[38]].write("%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n" % (line[10], line[0], line[0], line[48], label, line[12], line[13], line[14]))

  # close smiles files
  for k in abbr_dict.keys():
    outfile_dict[k].close()




if __name__ == "__main__":
  """
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  txt_dir = "txt_downloaded_from_chembl/"
  structure_dir = "structure_files/"
  if not os.path.exists(structure_dir):
    os.mkdir(structure_dir)

  for target in target_list:
    txt2smile(txt_dir + target + ".txt", structure_dir + target + ".smiles")
  """

  txt2smiles_1()
 



