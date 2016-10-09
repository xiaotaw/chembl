#!/usr/bin/python
# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: evaluate pk model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy
import datetime
import tensorflow as tf
import pk_model
import pk_input

def evaluate(target_list):
  """ evaluate the model 
  """
  # virtual screen log file
  log_dir = "log_files"
  logpath = os.path.join(log_dir, "pk_eval.log")
  logfile = open(logpath, "w")
  logfile.write("pk_eval starts at: %s\n" % datetime.datetime.now())

  # get input dataset
  train_dataset_dict = dict()
  test_dataset_dict = dict()
  for target in target_list:
    train_dataset_dict[target] = pk_input.get_inputs_by_cpickle("data_files/pkl_files/" + target + "_train.pkl") 
    test_dataset_dict[target] = pk_input.get_inputs_by_cpickle("data_files/pkl_files/" + target + "_test.pkl") 

  neg_dataset = pk_input.get_inputs_by_cpickle("data_files/pkl_files/pubchem_neg_sample.pkl")
  


  with tf.Graph().as_default(), tf.device("/gpu:0"):
    
    # build the model
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 8192))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # build the "Tree" with a mutual "Term" and several "Branches"
    base = pk_model.term(input_placeholder, keep_prob=1.0)
    softmax_dict = dict()
    wd_loss_dict = dict()
    x_entropy_dict = dict()
    loss_dict = dict()
    accuracy_dict = dict()
    for target in target_list:
      # compute softmax
      softmax_dict[target] = pk_model.branch(target, base, keep_prob=1.0)
      # compute loss.
      wd_loss_dict[target] = tf.add_n(tf.get_collection("term_wd_loss") + tf.get_collection(target+"_wd_loss"))
      x_entropy_dict[target] = pk_model.x_entropy(softmax_dict[target], label_placeholder, target)
      loss_dict[target]  = tf.add(wd_loss_dict[target], x_entropy_dict[target])
      # compute accuracy
      accuracy_dict[target] = pk_model.accuracy(softmax_dict[target], label_placeholder, target)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())

    # create session.
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)

    # Restores variables from checkpoint
    saver.restore(sess, "ckpt_files/model.ckpt-40000")

    
    
    # eval train dataset
    for target in target_list:
      t0 = float(time.time())
      compds = numpy.vstack([train_dataset_dict[target].compds, neg_dataset.compds])
      labels = numpy.vstack([train_dataset_dict[target].labels, neg_dataset.labels])
      t1 = float(time.time())
      LV, XLV, ACC, prediction, label_dense = sess.run(
        [wd_loss_dict[target], 
         x_entropy_dict[target],
         accuracy_dict[target], 
         tf.argmax(softmax_dict[target], 1), 
         tf.argmax(labels, 1)], 
        feed_dict = {
          input_placeholder: compds,
          label_placeholder: labels,
        }
      )
      t2 = time.time()
      TP, TN, FP, FN, SEN, SPE, MCC = pk_model.compute_performance(label_dense, prediction)
      format_str = "%6d %6d %6.3f %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %5.3f %5.3f %s"
      logfile.write(format_str % (5000, 40000, LV, XLV, 0, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))
      logfile.write('\n')
      print(format_str % (5000, 40000, LV, XLV, 0, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))  

    # eval test dataset
    for target in target_list:  
      t0 = float(time.time())
      compds = test_dataset_dict[target].compds
      labels = test_dataset_dict[target].labels
      t1 = float(time.time())
      LV, XLV, ACC, prediction, label_dense = sess.run(
        [wd_loss_dict[target], 
         x_entropy_dict[target],
         accuracy_dict[target], 
         tf.argmax(softmax_dict[target], 1), 
         tf.argmax(labels, 1)], 
        feed_dict = {
          input_placeholder: compds,
          label_placeholder: labels,
        }
      )
      t2 = time.time()
      TP, TN, FP, FN, SEN, SPE, MCC = pk_model.compute_performance(label_dense, prediction)
      format_str = "%6d %6d %6.3f %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %5.3f %5.3f %s"
      logfile.write(format_str % (5000, 40000, LV, XLV, 0, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))
      logfile.write('\n')
      print(format_str % (5000, 40000, LV, XLV, 0, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))  

  logfile.close()


if __name__ == "__main__":
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  evaluate(target_list)



