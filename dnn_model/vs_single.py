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

  # virtual screen log file
  log_dir = "log_files"
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  log_path = os.path.join(log_dir, "vs_%s_%d_%4.3f_%4.3e_%d.log" % (target, batch_size, keep_prob, wd, part_num))
  logfile = open(log_path, 'w')
  print("virtual screen %d starts at: %s\n" % (part_num, datetime.datetime.now()))

  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%4.3f_%4.3e.ckpt' % (batch_size, keep_prob, wd))

  # input and output dir
  fp_dir = "/raid/xiaotaw/pubchem/fp_files/%d" % part_num

  # screening
  #with tf.Graph().as_default(), tf.device("/gpu:%d" % (part_num // 2)):
  with tf.Graph().as_default(), tf.device("/gpu:%d" % (part_num % 4)):
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
      #for i in xrange(10000001, 12000001, 25000):
        in_file = "Compound_" + "{:0>9}".format(i) + "_" + "{:0>9}".format(i + 24999) + ".apfp"
        fp_fn = os.path.join(fp_dir, in_file)
        if not os.path.exists(fp_fn):
          print("%s not exists" % fp_fn)
          continue

        d.reset(fp_fn)
        compds = d.features_dense

        #pred = sess.run(tf.argmax(softmax, 1), feed_dict = {input_placeholder: compds})
        sm = sess.run(softmax, feed_dict = {input_placeholder: compds})
        
        for id_, sm_v in zip(d.pubchem_id, sm[:, 1]):
          logfile.writelines("%s\t%f\n" % (id_, sm_v))
        
        print("%s\t%d\n" % (fp_fn, len(d.pubchem_id)))

        #result = np.array(d.pubchem_id)[pred.astype(bool)]
        #logfile.writelines(["%s\n" % x for x in result])
        #print("%s\t%d\n" % (fp_fn, result.shape[0]))


  print("duration: %.3f" % (time.time() - t_0))


if __name__ == "__main__":
  target_list = ['CHEMBL203', 'CHEMBL204', 'CHEMBL205', 'CHEMBL214', 'CHEMBL217',
               'CHEMBL218', 'CHEMBL220', 'CHEMBL224', 'CHEMBL225', 'CHEMBL226',
               'CHEMBL228', 'CHEMBL230', 'CHEMBL233', 'CHEMBL234', 'CHEMBL235',
               'CHEMBL236', 'CHEMBL237', 'CHEMBL240', 'CHEMBL244', 'CHEMBL251',
               'CHEMBL253', 'CHEMBL256', 'CHEMBL259', 'CHEMBL260', 'CHEMBL261',
               'CHEMBL264', 'CHEMBL267', 'CHEMBL279', 'CHEMBL284', 'CHEMBL2842',
               'CHEMBL289', 'CHEMBL325', 'CHEMBL332', 'CHEMBL333', 'CHEMBL340',
               'CHEMBL344', 'CHEMBL4005', 'CHEMBL4296', 'CHEMBL4722', 'CHEMBL4822']

  # the target
  target = "CHEMBL204"

  # part_num range from 0 to 12(included)
  virtual_screening_single(target, int(sys.argv[1]))



