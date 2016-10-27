# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: define functions and parameters related to input data


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import time
import random

import numpy as np

from scipy import sparse

vec_len = 2039
data_dir = "../data_files"
h5_dir = os.path.join(data_dir, "h5_files")


def dense_to_one_hot(labels_dense, num_classes=2, dtype=np.int):
  """Convert class labels from scalars to one-hot vectors.
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel().astype(dtype)] = 1
  return labels_one_hot


class Dataset(object):
  def __init__(self, size, is_shuffle=False, fold=10):

    self.size = size
    self.perm = np.array(range(self.size))
    if is_shuffle:
      random.shuffle(self.perm)
 
    self.train_size = int(self.size * (1.0 - 1.0 / fold))
    self.train_perm = self.perm[range(self.train_size)]
    self.train_begin = 0
    self.train_end = 0

    self.test_perm = self.perm[range(self.train_size, self.size)]

  def generate_perm_for_train_batch(self, batch_size):
    self.train_begin = self.train_end
    self.train_end += batch_size
    if self.train_end > self.train_size:
      random.shuffle(self.train_perm)
      self.train_begin = 0
      self.train_end = batch_size
    perm = self.train_perm[self.train_begin: self.train_end]
    return perm


class PosDataset(Dataset):
  def __init__(self, target, one_hot=True, dtype=np.float32):
    # open h5 file
    self.h5_fn = os.path.join(h5_dir, target + ".h5")
    self.h5 = h5py.File(self.h5_fn, "r")
    # read ids
    self.ids = self.h5["chembl_id"].value
    # read 3 fp, and stack as feauture
    ap = sparse.csr_matrix((self.h5["ap"]["data"], self.h5["ap"]["indices"], self.h5["ap"]["indptr"]), shape=[len(self.h5["ap"]["indptr"]) - 1, vec_len])
    mg = sparse.csr_matrix((self.h5["mg"]["data"], self.h5["mg"]["indices"], self.h5["mg"]["indptr"]), shape=[len(self.h5["mg"]["indptr"]) - 1, vec_len])
    tt = sparse.csr_matrix((self.h5["tt"]["data"], self.h5["tt"]["indices"], self.h5["tt"]["indptr"]), shape=[len(self.h5["tt"]["indptr"]) - 1, vec_len])
    self.features = sparse.hstack([ap, mg, tt]).toarray()
    # label 
    self.labels = self.h5["label"].value
    if one_hot == True:
      self.labels = dense_to_one_hot(self.labels)
    # year
    if "year" in self.h5.keys():
      self.years = self.h5["year"].value
    else:
      self.years = None
    # close h5 file
    self.h5.close()
    # dtype
    self.dtype = dtype
    # pre_process
    #self.features = np.log10(1.0 + self.features).astype(self.dtype)
    self.features = np.clip(self.features, 0, 1).astype(self.dtype)
    # 
    Dataset.__init__(self, self.features.shape[0])


  def next_train_batch(self, batch_size):
    perm = self.generate_perm_for_train_batch(batch_size)
    return self.features[perm], self.labels[perm]


class NegDataset(Dataset):
  def __init__(self, target_list, one_hot=True, dtype=np.float32):
    # open h5 file
    self.h5_fn = os.path.join(h5_dir, "pubchem_neg_sample.h5")
    self.h5 = h5py.File(self.h5_fn, "r")
    # read ids
    self.ids = self.h5["chembl_id"].value
    # read 3 fp, and stack as feauture
    ap = sparse.csr_matrix((self.h5["ap"]["data"], self.h5["ap"]["indices"], self.h5["ap"]["indptr"]), shape=[len(self.h5["ap"]["indptr"]) - 1, vec_len])
    mg = sparse.csr_matrix((self.h5["mg"]["data"], self.h5["mg"]["indices"], self.h5["mg"]["indptr"]), shape=[len(self.h5["mg"]["indptr"]) - 1, vec_len])
    tt = sparse.csr_matrix((self.h5["tt"]["data"], self.h5["tt"]["indices"], self.h5["tt"]["indptr"]), shape=[len(self.h5["tt"]["indptr"]) - 1, vec_len])
    self.features = sparse.hstack([ap, mg, tt]).toarray()
    # label(mask)
    self.mask_dict = {}
    for target in target_list:
      #mask = self.h5["mask"][target].value
      mask = self.h5["cliped_mask"][target].value
      if one_hot == True:
        self.mask_dict[target] = dense_to_one_hot(mask)
      else:
        self.mask_dict[target] = mask
    # close h5 file
    self.h5.close()
    # dtype
    self.dtype = dtype
    # pre_process
    #self.features = np.log10(1.0 + self.features).astype(self.dtype)
    self.features = np.clip(self.features, 0, 1).astype(self.dtype)
    # 
    Dataset.__init__(self, self.features.shape[0])

  def next_train_batch(self, target, batch_size):
    perm = self.generate_perm_for_train_batch(batch_size)
    return self.features[perm], self.mask_dict[target][perm]


class Datasets(object):
  """dataset class, contains compds and labels, 
     and also provides method to generate a next batch of data. 
  """
  def __init__(self, target_list, one_hot=False):

    self.neg = NegDataset(target_list, one_hot=one_hot)

    # read pos sample 
    # and generate mask for neg sample(it's very complex, and I am not going to tell you the details)
    self.pos = {}
    for target in target_list:
      self.pos[target] = PosDataset(target, one_hot=one_hot)

  def next_train_batch(self, target, pos_batch_size, neg_batch_size):
    pos_feature_batch, pos_label_batch = self.pos[target].next_train_batch(pos_batch_size)
    neg_feature_batch, neg_label_batch = self.neg.next_train_batch(target, neg_batch_size)
    return np.vstack([pos_feature_batch, neg_feature_batch]), np.vstack([pos_label_batch, neg_label_batch])
    


""" test
import pk_input as pki
target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]
d = pki.Datasets(target_list)

for step in range(20 * 500):
  for target in target_list:
    compds_batch, labels_batch = d.next_train_batch(target, 128, 128)
    if np.isnan(compds_batch).sum() > 0 or (step % 500) == 0:
      print("%7d %3.2f %3.2f %3.2f %3.2f" % (step, compds_batch.min(), compds_batch.max(), labels_batch.min(), labels_batch.max()))

"""


if __name__ == "__main__":
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]








