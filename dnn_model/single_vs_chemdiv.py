

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


def virtual_screening_chemdiv(target, g_step, gpu_num=0):
  t_0 = time.time()
  
  # dataset
  d = ci.DatasetChemDiv(target)
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
  pred_path = os.path.join(pred_dir, "vs_chemdiv_%s_%d_%4.3f_%4.3e_%d.pred" % (target, batch_size, keep_prob, wd, g_step))
  predfile = open(pred_path, 'w')
  print("virtual screen ChemDiv starts at: %s\n" % datetime.datetime.now())

  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%4.3f_%4.3e.ckpt' % (batch_size, keep_prob, wd))

  # screening
  with tf.Graph().as_default(), tf.device("/gpu: %d" % gpu_num):
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
      try:     
        while True:
          ids, features = d.generate_batch(batch_size)
          sm = sess.run(softmax, feed_dict = {input_placeholder: features.toarray()})
          for id_, sm_v in zip(ids, sm[:, 1]):
            predfile.write("%s\t%f\n" % (id_, sm_v))
      except StopIteration:
        pass

  predfile.close()
  print("duration: %.3f" % (time.time() - t_0))


virtual_screening_chemdiv(target="CHEMBL203", g_step=2161371, gpu_num=0)






