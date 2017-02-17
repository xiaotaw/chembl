# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Nov 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description:

import os
import getpass
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict

data_dir = "/home/%s/Documents/chembl/data_files/" % getpass.getuser()
fp_dir = os.path.join(data_dir, "fp_files")
mask_dir = os.path.join(data_dir, "mask_files")
structure_dir = os.path.join(data_dir, "structure_files")

# the newly picked out 15 targets, include 9 targets from 5 big group, and 6 targets from others.
target_list = ["CHEMBL279", "CHEMBL203", # Protein Kinases
               "CHEMBL217", "CHEMBL253", # GPCRs (Family A)
               "CHEMBL235", "CHEMBL206", # Nuclear Hormone Receptors
               "CHEMBL240", "CHEMBL4296", # Voltage Gated Ion Channels
               "CHEMBL4805", # Ligand Gated Ion Channels
               "CHEMBL204", "CHEMBL244", "CHEMBL4822", "CHEMBL340", "CHEMBL205", "CHEMBL4005" # Others
              ] 


def dense_to_one_hot(labels_dense, num_classes=2, dtype=np.int):
  """Convert class labels from scalars to one-hot vectors.
  Args:
    labels_dense: <type 'numpy.ndarray'> dense label
    num_classes: <type 'int'> the number of classes in one hot label
    dtype: <type 'type'> data type
  Return:
    labels_ont_hot: <type 'numpy.ndarray'> one hot label
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel().astype(dtype)] = 1
  return labels_one_hot


def sparse_features(fps_list, target_columns_dict, num_features, is_log=True):
  """construct a sparse matrix(csr_matrix) for features according to target_columns_dict.
  Args:
    fps_list: <type 'list'> a list of apfps for the molecules
    is_log: <type 'bool'> flag whether apply np.log to data, default is True
  Return:
    features: the sparse matrix of features
  """
  data = []
  indices = []
  indptr = [0]
  for fps_str in fps_list:
    n = indptr[-1]
    for fp in fps_str[1:-1].split(","):
      if ":" in fp:
        k, v = fp.split(":")
        indices.append(target_columns_dict[int(k)])
        data.append(int(v))
        n += 1
    indptr.append(n)
  data = np.array(data)
  if is_log:
    data = np.log(data).astype(np.float32)
  # here we add one to num_features, because any apfp not founded in target_apfp_picked will be mapped
  # to the last column of the features matrix, though the last column will not be used ultimately.
  features = sparse.csr_matrix((data, indices, indptr), shape=(len(fps_list), num_features + 1))
  return features


class DatasetBase(object):
  def __init__(self, target):
    # read count and the apfps that were picked out 
    counts = np.genfromtxt(mask_dir + "/%s_apfp.count" % target, delimiter="\t", dtype=int)
    self.target_apfp_picked = counts[counts[:, 1] > 10][:, 0]
    self.target_apfp_picked.sort()
    self.num_features = len(self.target_apfp_picked)
    # columns and sparse features
    # here we use a defaultdict, where any apfp not founded in target_apfp_picked will be mapped
    # to the last column of the features matrix, though the last column will not be used ultimately.
    self.target_columns_dict = defaultdict(lambda : self.num_features)
    for i, apfp in enumerate(self.target_apfp_picked):
      self.target_columns_dict[apfp] = i

  def batch_generator_base(self, size, batch_size):
    begin = 0
    end = 0
    while True:
      begin = end
      if begin >= size:
        raise StopIteration()
      end += batch_size 
      if end > size:
        end = size
      yield begin, end
    

class DatasetTarget(DatasetBase):
  def __init__(self, target, year_split=2014):
    DatasetBase.__init__(self, target)
    # read chembl id and apfp
    self.chembl_id = []
    self.chembl_apfp = {}
    f = open(fp_dir + "/chembl.apfp", "r")
    for line in f:
      id_, fps_str = line.split("\t")
      id_ = id_.strip()
      fps_str = fps_str.strip()
      self.chembl_id.append(id_)
      self.chembl_apfp[id_] = fps_str
    f.close()
    # read top 79 targets' label data, and get the specific target's label data
    clf_label_79 = pd.read_csv(structure_dir + "/chembl_top79.label", usecols=[0, 2, 3, 4], delimiter="\t")
    self.target_clf_label = clf_label_79[clf_label_79["TARGET_CHEMBLID"] == target]
    # remove compounds whose apfp cannot be caculated
    m = self.target_clf_label["CMPD_CHEMBLID"].isin(self.chembl_id)
    self.target_clf_label = self.target_clf_label[m.values] 
    # time split
    time_mask = self.target_clf_label["YEAR"] > year_split
    time_split_train = self.target_clf_label[~time_mask] 
    time_split_test = self.target_clf_label[time_mask]
    # ids
    self.target_ids_train = time_split_train["CMPD_CHEMBLID"].values
    self.target_ids_test = time_split_test["CMPD_CHEMBLID"].values
    # features   
    self.target_features_train = sparse_features([self.chembl_apfp[k] for k in self.target_ids_train], self.target_columns_dict, self.num_features)[:, :-1]
    self.target_features_test = sparse_features([self.chembl_apfp[k] for k in self.target_ids_test], self.target_columns_dict, self.num_features)[:, :-1]
    # labels
    self.target_labels_train = (time_split_train["CLF_LABEL"] > 0).astype(int).values
    self.target_labels_test = (time_split_test["CLF_LABEL"] > 0).astype(int).values


class DatasetCNS(DatasetTarget):
  def __init__(self, target, year_split=2014):
    DatasetTarget.__init__(self, target, year_split=year_split)
    # read mask
    self.cns_mask = pd.Series.from_csv(mask_dir + "/%s_cns.mask" % target, header=None, sep="\t")    
    # features
    self.cns_features = sparse_features([self.chembl_apfp[k] for k in self.chembl_id], self.target_columns_dict, self.num_features)[:, :-1]
    # 
    m = self.cns_mask.index.isin(self.target_ids_test)
    self.cns_features_train = self.cns_features[~m]
    self.cns_mask_train = self.cns_mask[~m]

  def batch_generator_cns(self, batch_size):
    for begin, end in self.batch_generator_base(self.cns_features.shape[0], batch_size):
      ids = self.chembl_id[begin: end]
      features = self.cns_features[begin: end].toarray()
      mask = self.cns_mask[begin: end].values
      yield ids, features, mask


class DatasetPNS(DatasetBase):
  def __init__(self, target):
    DatasetBase.__init__(self, target)
    # read pns apfp
    self.pns_id = []
    self.pns_apfp = {}
    f = open(fp_dir + "/pubchem_neg_sample.apfp", "r")
    for line in f:
      id_, fps_str = line.split("\t")
      id_ = id_.strip()
      fps_str = fps_str.strip()
      self.pns_id.append(id_)
      self.pns_apfp[id_] = fps_str                
    f.close()
    # read mask
    self.pns_mask = pd.Series.from_csv(mask_dir + "/%s_pns.mask" % target, header=None, sep="\t")
    # features
    self.pns_features = sparse_features([self.pns_apfp[k] for k in self.pns_id], self.target_columns_dict, self.num_features)[:, :-1]

  def batch_generator_pns(self, batch_size):
    for begin, end in self.batch_generator_base(self.pns_features.shape[0], batch_size):
      ids = self.pns_id[begin: end]
      features = self.pns_features[begin: end].toarray()
      mask = self.pns_mask[begin: end].values
      yield ids, features, mask


class Dataset(DatasetCNS, DatasetPNS):
  """Base dataset class for chembl inhibitors
  """
  def __init__(self, target, one_hot=True, is_shuffle_train=True,  train_pos_multiply=0):
    """Constructor, create a dataset container. 
    Args:
      target: <type 'str'> the chemblid of the target, e.g. "CHEMBL203".
      one_hot: <type 'bool'> flag whether create one_hot label, default is True.
      is_shuffle: <type 'bool'> flag whether shuffle samples when the dataset created.
      year_split: <type 'int'> time split year, 
        if a molecule's year > year_split, it will be split into test data,
        otherwise, if a molecule's year <= year_split, it will be split into train data.
    Return:
      None
    """
    DatasetCNS.__init__(self, target, year_split=2014)
    DatasetPNS.__init__(self, target)
    # cns train pos 
    self.cns_features_train_pos = self.cns_features_train[self.cns_mask_train.values]
    self.cns_mask_train_pos = self.cns_mask_train[self.cns_mask_train.values]
    # train, if train_pos_multiply > 0, cns_train_pos will be extra added for train_pos_multiply times .
    tf_list = [self.cns_features_train, self.pns_features]
    tl_list = [self.cns_mask_train, self.pns_mask]
    for _ in range(train_pos_multiply):
      tf_list.append(self.cns_features_train_pos)
      tl_list.append(self.cns_mask_train_pos)
    self.train_features = sparse.vstack(tf_list)
    self.train_labels = np.hstack(tl_list).astype(int)
    # test
    self.test_features = self.target_features_test
    self.test_labels = self.target_labels_test
    # one_hot
    if one_hot:
      self.train_labels_one_hot = dense_to_one_hot(self.train_labels) 
      self.test_labels_one_hot = dense_to_one_hot(self.test_labels) 
    # batch related
    self.train_size = self.train_features.shape[0] # (954049, 9412)
    self.train_perm = np.array(range(self.train_size))
    if is_shuffle_train:
      np.random.shuffle(self.train_perm)
    self.train_begin = 0
    self.train_end = 0

    self.cns_size = self.cns_features.shape[0] # (878721, 9412)
    self.cns_perm = np.arange(self.cns_size)
    self.cns_begin = 0
    self.cns_end = 0

  def generate_perm_for_train_batch(self, batch_size):
    """Create the permutation for a batch of train samples
    Args:
      batch_size: <type 'int'> the number of samples in the batch
    Return:
      perm: <type 'numpy.ndarray'> the permutation of samples which form a batch
    """
    self.train_begin = self.train_end
    self.train_end += batch_size
    if self.train_end > self.train_size:
      np.random.shuffle(self.train_perm)
      self.train_begin = 0
      self.train_end = batch_size
    perm = self.train_perm[self.train_begin: self.train_end]
    return perm

  def generate_train_batch(self, batch_size):
    perm = self.generate_perm_for_train_batch(batch_size)
    return self.train_features[perm].toarray().astype(np.float32), self.train_labels_one_hot[perm]

  def reset_begin_end(self):
    self.train_begin = 0
    self.train_end = 0

  def generate_train_batch_once(self, batch_size):
    self.train_begin = self.train_end
    self.train_end += batch_size
    if self.train_end > self.train_size:
      self.train_end = self.train_size
    perm = self.train_perm[self.train_begin: self.train_end]
    return self.train_features[perm].toarray().astype(np.float32), self.train_labels_one_hot[perm]



# dataset for virtual screening(vs)
class DatasetVS(DatasetBase):
  def __init__(self, target):
    DatasetBase.__init__(self, target)

  def reset(self, fp_fn):
    # read chembl id and apfp
    self.pubchem_id = []
    self.pubchem_apfp = {}
    f = open(fp_fn, "r")
    for line in f:
      id_, fps_str = line.split("\t")
      id_ = id_.strip()
      fps_str = fps_str.strip()
      self.pubchem_id.append(id_)
      self.pubchem_apfp[id_] = fps_str
    f.close()
    # generate features
    self.features = sparse_features([self.pubchem_apfp[k] for k in self.pubchem_id], self.target_columns_dict, self.num_features)[:, :-1]
    self.features_dense = self.features.toarray()


class DatasetChemDiv(DatasetBase):
  def __init__(self, target):
    DatasetBase.__init__(self, target)
    # read ids and apfps
    ChemDiv_dir = "/raid/xiaotaw/ChemDiv"
    fn_list = ["DC01_350000.apfp", "DC02_350000.apfp", "DC03_222773.apfp", "DC_saltdata_not-available_124145.apfp", "IC_non-excl_82693.apfp", "NC_340320.apfp"]
    self.chemdiv_ids = []
    self.chemdiv_apfps = {}
    for fn in fn_list:
      f = open(ChemDiv_dir + "/" + fn, "r")
      for line in f:
        id_, fps_str = line.split("\t")
        id_ = id_.strip()
        fps_str = fps_str.strip()
        self.chemdiv_ids.append(id_)
        self.chemdiv_apfps[id_] = fps_str                
      f.close()
    # batch related
    self.begin = 0
    self.end = 0
    self.size = len(self.chemdiv_ids)
  
  def generate_batch(self, batch_size):
    self.begin = self.end
    if self.begin >= self.size:
      raise StopIteration()
    self.end += batch_size
    if self.end > self.size:
      self.end = self.size
    ids = self.chemdiv_ids[self.begin: self.end]
    apfp_list = [self.chemdiv_apfps[k] for k in ids]
    features = sparse_features(apfp_list, self.target_columns_dict, self.num_features)[:, :-1]
    return ids, features



def compute_performance(label, prediction):
  """sensitivity(SEN), specificity(SPE), accuracy(ACC), matthews correlation coefficient(MCC) 
  """
  assert label.shape[0] == prediction.shape[0], "label number should be equal to prediction number"
  N = label.shape[0]
  APP = sum(prediction)
  ATP = sum(label)
  TP = sum(prediction * label)
  FP = APP - TP
  FN = ATP - TP
  TN = N - TP - FP - FN
  SEN = float(TP) / (ATP) if ATP != 0 else np.nan
  SPE = float(TN) / (N - ATP)
  ACC = float(TP + TN) / N
  MCC = (TP * TN - FP * FN) / (np.sqrt(long(N - APP) * long(N - ATP) * APP * ATP)) if not (N - APP) * (N - ATP) * APP * ATP == 0 else 0.0
  return TP, TN, FP, FN, SEN, SPE, ACC, MCC








