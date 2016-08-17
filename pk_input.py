# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: define functions and parameters related to input data


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
import cPickle
import random # for random.shuffle()

data_dir = "data_files"
mgfp_dir = "mgfp_files" 

class DataSet(object):
  """dataset class, contains compds and labels, 
     and also provides method to generate a next batch of data. 
  """
  def __init__(self, compds, labels, dtype=numpy.float32):
    assert compds.shape[0] == labels.shape[0], "shape don't match"
    self.compds = compds.astype(dtype)
    self.labels = labels
    self.size = compds.shape[0]
    self.begin = 0
    self.end = 0

  def generate_batch(self, batch_size):
    assert self.compds.shape[0] == self.labels.shape[0], "shape don't match"
    assert batch_size <= self.size, "too big batch_size"
    self.end = self.begin + batch_size
    if self.end > self.size:
      self.shuffle()
      self.begin = 0
      self.end = batch_size
    compds_batch = self.compds[self.begin: self.end]
    labels_batch = self.labels[self.begin: self.end]
    self.begin = self.end
    return compds_batch, labels_batch
  
  def shuffle(self):
    perm = range(self.size)
    random.shuffle(perm)
    self.compds = self.compds[perm]
    self.labels = self.labels[perm]

def dense_to_one_hot(labels_dense, num_classes=2):
  """Convert class labels from scalars to one-hot vectors.
  """
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel().astype(int)] = 1
  # print(labels_one_hot)
  return labels_one_hot

def get_inputs(target):
  """read dataset from file according to target name
     note: target.compds are 2048 length mgfp code files, which were generated by 'cPickle.dump'.
  Args:
    target: a string of the protein kinase target name
  """
  with open(os.path.join(data_dir, target + ".compds"), "r") as compds_file, open(os.path.join(data_dir, target + ".labels"), "r") as labels_file:
    compds = cPickle.load(compds_file)
    labels = cPickle.load(labels_file)
    return DataSet(compds, labels, )

def get_inputs_pseudo(target, is_pos):
  """read dataset from file according to target name,
     'pseudo' means that labels are manually generated rather than read from file.
     note: target.compds are 2048 length mgfp code files, which were generated by 'cPickle.dump'.
  Args:
    target: a string of the protein kinase target name.
    is_pos: if true, labels are ones, else labels are zeros.
  """
  with open(os.path.join(data_dir, target + ".compds"), "r") as compds_file:
    compds = cPickle.load(compds_file)
    numpy.clip(compds, 0, 1, out=compds)
    if is_pos:
      labels_dense = numpy.ones(compds.shape[0], dtype=numpy.int32)
    else:
      labels_dense = numpy.zeros(compds.shape[0], dtype=numpy.int32)
    labels = dense_to_one_hot(labels_dense)
    return DataSet(compds, labels)

def 

