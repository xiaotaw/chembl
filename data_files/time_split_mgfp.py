#!/usr/bin/python
# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: time split for chembl data. 
#              based on mgfp files

mgfp_dir = "mgfp_files/"


def time_split(target, split_rate):
  
  # get all the molecules' info
  infile = open(mgfp_dir + target + ".mgfp6", "r")
  lines = infile.readlines()
  infile.close()

  # analyse year distribution, get the year_list like:
  # [(1999, 100), ..., (year, num_molecule_this_year)]
  year_dict = {}
  for line in lines:
    year = line.split("\t")[3]
    if year_dict.has_key(year):
      year_dict[year] += 1
    else:
      year_dict[year] = 1

  # sorted() has return value, while list.sort() has not
  year_list = sorted([(x, year_dict[x]) for x in year_dict.keys()], key=lambda x: x[0][:-1])

  # get the year which belongs to train_data
  min_train_num = len(lines) * split_rate
  count = 0
  train_year = []
  for year, number in year_list:
    count += number;
    train_year.append(year)
    if count >= min_train_num:
      break

  # out put into file
  outfile_train = open(mgfp_dir + target + "_train.mgfp6", "w")
  outfile_test = open(mgfp_dir + target + "_test.mgfp6", "w")

  for line in lines:
    year = line.split("\t")[3]
    if year in train_year:
      outfile_train.write(line)
    else:
      outfile_test.write(line)

  outfile_train.close()
  outfile_test.close()

  print("%s: total %d train %d test %d " % (target, len(lines), count, len(lines) - count))



if __name__ == "__main__":

  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  # time split by year, and ensure:
  # train >= total * split_rate
  # test = total - train
  split_rate = 0.8

  for target in target_list:
    time_split(target, split_rate)

  
