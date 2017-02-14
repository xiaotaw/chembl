#!/usr/bin/python
# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: apply pk model to pubchem dataset, to screen potential active substrate(drugs)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import datetime
import tensorflow as tf
from matplotlib import pyplot as plt

import dnn_model
sys.path.append("/home/scw4750/Documents/chembl/data_files/")
import chembl_input as ci





def virtual_screening_single(target, g_step, part_num, gpu_num):
  t_0 = time.time()

  # dataset
  d = ci.DatasetVS(target)
  # batch size
  batch_size = 128
  # input vec_len
  input_vec_len = d.num_features
  # keep prob
  keep_prob = 0.8
  # weight decay
  wd = 0.004
  # g_step
  #g_step = 2236500 

  # virtual screen pred file
  pred_dir = "pred_files/%s" % target
  if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
  pred_path = os.path.join(pred_dir, "vs_pubchem_%s_%d_%4.3f_%4.3e_%d_%d.pred" % (target, batch_size, keep_prob, wd, g_step, part_num))
  predfile = open(pred_path, 'w')
  print("virtual screen %d starts at: %s\n" % (part_num, datetime.datetime.now()))

  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%4.3f_%4.3e.ckpt' % (batch_size, keep_prob, wd))

  # input and output dir
  fp_dir = "/raid/xiaotaw/pubchem/fp_files/%d" % part_num

  # screening
  with tf.Graph().as_default(), tf.device("/gpu: %d" % gpu_num):
  #with tf.Graph().as_default(), tf.device("/gpu:%d" % (part_num % 4)):
    # the input
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))
    # the term
    base = dnn_model.term(input_placeholder, in_units=input_vec_len, wd=wd, keep_prob=1.0)
    # the branches
    softmax = dnn_model.branch(target, base, wd=wd, keep_prob=1.0)
    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())
    # Start screen
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.35
    with tf.Session(config=config) as sess:
      # Restores variables from checkpoint
      saver.restore(sess, ckpt_path + "-%d" % g_step)
      for i in xrange(part_num * 10000000 + 1, (part_num + 1) * 10000000, 25000):
        in_file = "Compound_" + "{:0>9}".format(i) + "_" + "{:0>9}".format(i + 24999) + ".apfp"
        fp_fn = os.path.join(fp_dir, in_file)
        if not os.path.exists(fp_fn):
          print("%s not exists" % fp_fn)
          continue
        d.reset(fp_fn)
        compds = d.features_dense
        sm = sess.run(softmax, feed_dict = {input_placeholder: compds})
        for id_, sm_v in zip(d.pubchem_id, sm[:, 1]):
          predfile.writelines("%s\t%f\n" % (id_, sm_v))
        print("%s\t%d\n" % (fp_fn, len(d.pubchem_id)))

  print("duration: %.3f" % (time.time() - t_0))



def predict(target, g_step_list=None):
  """ evaluate the model 
  """
  # dataset
  d = ci.Dataset(target)
  # batch size
  batch_size = 128
  # learning rate 
  step_per_epoch = int(d.train_size / batch_size)
  # input vec_len
  input_vec_len = d.train_features.shape[1]
  # keep prob
  keep_prob = 0.8
  # weight decay
  wd = 0.004
  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%4.3f_%4.3e.ckpt' % (batch_size, keep_prob, wd))
  # pred file
  pred_dir = "pred_files/%s" % target
  if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
  
  # g_step_list
  #g_step_list = range(1, 2235900, 10 * step_per_epoch)
  #g_step_list = [2161371]

  with tf.Graph().as_default(), tf.device("/gpu:3"):
    
    # build the model
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # build the "Tree" with a mutual "Term" and several "Branches"
    base = dnn_model.term(input_placeholder, in_units=input_vec_len, wd=wd, keep_prob=1.0)
    # compute softmax
    softmax = dnn_model.branch(target, base, wd=wd, keep_prob=1.0)
    # compute loss.
    wd_loss = tf.add_n(tf.get_collection("term_wd_loss") + tf.get_collection(target+"_wd_loss"))
    x_entropy = dnn_model.x_entropy(softmax, label_placeholder, target)
    loss  = tf.add(wd_loss, x_entropy)
    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())
    # create session.
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)

    # target test
    test_chemblid = d.time_split_test["CMPD_CHEMBLID"]
    test_compds = d.test_features.toarray()
    test_labels_dense = d.test_labels

    # target train
    time_split_train = d.target_clf_label[d.target_clf_label["YEAR"] <= 2014]
    target_train_chemblid = time_split_train["CMPD_CHEMBLID"]
    m = d.target_cns_mask.index.isin(time_split_train["CMPD_CHEMBLID"])
    target_train_features = d.target_cns_features[m].toarray()
    target_train_labels_dense = d.target_cns_mask[m].values.astype(int)

    for g_step in g_step_list:
      # Restores variables from checkpoint
      saver.restore(sess, ckpt_path + "-%d" % g_step)

      # the target's test
      sm = sess.run(softmax, feed_dict = {input_placeholder: test_compds})

      test_pred_path = os.path.join(pred_dir, "test_%s_%d_%4.3f_%4.3e_%d.pred" % (target, batch_size, keep_prob, wd, g_step))
      test_pred_file = open(test_pred_path, 'w')

      for id_, sm_v, l_v in zip(test_chemblid, sm[:, 1], test_labels_dense):
        test_pred_file.writelines("%s\t%f\t%f\n" % (id_, sm_v, l_v))

      test_pred_file.close()

      # the target's train
      sm = sess.run(softmax, feed_dict = {input_placeholder: target_train_features})

      train_pred_path = os.path.join(pred_dir, "train_%s_%d_%4.3f_%4.3e_%d.pred" % (target, batch_size, keep_prob, wd, g_step))
      train_pred_file = open(train_pred_path, 'w')

      for id_, sm_v, l_v in zip(target_train_chemblid, sm[:, 1], target_train_labels_dense):
        train_pred_file.writelines("%s\t%f\t%f\n" % (id_, sm_v, l_v))

      train_pred_file.close()


def analyse(target, g_step):
  vs_pred_file = "pred_files/%s/vs_pubchem_%s_128_0.800_4.000e-03_%d.pred" % (target, target, g_step)
  aa = np.genfromtxt(vs_pred_file, delimiter="\t")
  a = aa[:, 1]

  test_pred_file = "pred_files/%s/test_%s_128_0.800_4.000e-03_%d.pred" % (target, target, g_step)
  bb = np.genfromtxt(test_pred_file, delimiter="\t", usecols=[1,2])

  b = bb[:, 0][bb[:, 1].astype(bool)]

  x = []
  y = []
  for i in range(10):
    mark = (i + 1) / 20.0
    xi = 1.0 * (b > mark).sum() / b.shape[0]
    yi = (a > mark).sum()
    x.append(xi)
    y.append(yi)

  plt.plot(x, y, "*")
  plt.xlabel("pos yeild rate")
  plt.ylabel("vs pubchem false pos")

  plt.savefig("pred_files/%s/%d.png" % (target, g_step))


  


if __name__ == "__main__":
  # the newly picked out 15 targets, include 9 targets from 5 big group, and 6 targets from others.
  target_list = ["CHEMBL279", "CHEMBL203", # Protein Kinases
               "CHEMBL217", "CHEMBL253", # GPCRs (Family A)
               "CHEMBL235", "CHEMBL206", # Nuclear Hormone Receptors
               "CHEMBL240", "CHEMBL4296", # Voltage Gated Ion Channels
               "CHEMBL4805", # Ligand Gated Ion Channels
               "CHEMBL204", "CHEMBL244", "CHEMBL4822", "CHEMBL340", "CHEMBL205", "CHEMBL4005" # Others
              ] 

  # the target
  target = "CHEMBL205"

  # part_num range from 0 to 12(included)
  #for i in range(9, 13):
  #  virtual_screening_single(target, 2161081, i, 3)

  predict(target, g_step_list=[2161081])

  analyse(target, g_step=2161081)



