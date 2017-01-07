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
import math
import glob
import time
import numpy as np
import datetime
import tensorflow as tf
from scipy import sparse

import pk_model
sys.path.append("/home/scw4750/Documents/chembl/data_files/")
import chembl_input as ci





def virtual_screening_single(target, part_num):
  t_0 = time.time()

  # dataset
  d = ci.DatasetVS(target)
  # batch size
  batch_size = 128
  # input vec_len
  input_vec_len = d.target_apfp_picked.shape[0]
  # keep prob
  keep_prob = 0.8
  # weight decay
  wd = 0.004
  # g_step
  g_step = 2236500 

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
  with tf.Graph().as_default(), tf.device("/gpu: 3"):
  #with tf.Graph().as_default(), tf.device("/gpu:%d" % (part_num % 4)):
    # the input
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))

    # the term
    base = pk_model.term(input_placeholder, in_units=input_vec_len, wd=wd, keep_prob=1.0)

    # the branches
    softmax = pk_model.branch(target, base, wd=wd, keep_prob=1.0)

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
  target = "CHEMBL204"

  # part_num range from 0 to 12(included)
  virtual_screening_single(target, int(sys.argv[1]))



