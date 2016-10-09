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

def evaluate(target, g_step):
  """ evaluate the model 
  """
  #
  log_dir = "log_files"
  ckpt_dir = os.path.join("ckpt_files", target)

  # eval log file
  log_path = os.path.join(log_dir, target + "_eval_single.log")
  logfile = open(log_path, 'w')
  logfile.write("eval starts at: %s\n" % datetime.datetime.now())

  # get input dataset
  pos_dataset = pk_input.get_inputs_by_cpickle("data_files/pkl_files/" + target + "_train.pkl", clip=True)
  test_dataset = pk_input.get_inputs_by_cpickle("data_files/pkl_files/" + target + "_test.pkl", clip=True)
  neg_dataset = pk_input.get_inputs_by_cpickle("data_files/pkl_files/pubchem_neg_sample.pkl", clip=True)

  with tf.Graph().as_default(), tf.device("/gpu:0"):
    
    # build the model
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 8192))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # build the "Tree" with a mutual "Term" and several "Branches"
    base = pk_model.term(input_placeholder, keep_prob=1.0)
    # compute softmax
    softmax = pk_model.branch(target, base, keep_prob=1.0)
    # compute loss.
    wd_loss = tf.add_n(tf.get_collection("term_wd_loss") + tf.get_collection(target+"_wd_loss"))
    x_entropy = pk_model.x_entropy(softmax, label_placeholder, target)
    loss  = tf.add(wd_loss, x_entropy)
    # compute accuracy
    accuracy = pk_model.accuracy(softmax, label_placeholder, target)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())

    # create session.
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)

    # Restores variables from checkpoint
    saver.restore(sess, ckpt_dir +"/model.ckpt-"+str(g_step))

    
    
    # eval train dataset
    t0 = float(time.time())
    compds = numpy.vstack([pos_dataset.compds, neg_dataset.compds])
    labels = numpy.vstack([pos_dataset.labels, neg_dataset.labels])
    t1 = float(time.time())
    LV, XLV, ACC, prediction, label_dense = sess.run(
      [wd_loss, 
       x_entropy,
       accuracy, 
       tf.argmax(softmax, 1), 
       tf.argmax(labels, 1)], 
      feed_dict = {
        input_placeholder: compds,
        label_placeholder: labels,
      }
    )
    t2 = time.time()
    TP, TN, FP, FN, SEN, SPE, MCC = pk_model.compute_performance(label_dense, prediction)
    format_str = "%6d %6d %6.3f %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %5.3f %5.3f %s"
    logfile.write(format_str % (g_step, g_step, LV, XLV, 0, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))
    logfile.write('\n')
    print(format_str % (g_step, g_step, LV, XLV, 0, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))  

    # eval test dataset  
    t0 = float(time.time())
    compds = test_dataset.compds
    labels = test_dataset.labels
    t1 = float(time.time())
    LV, XLV, ACC, prediction, label_dense = sess.run(
      [wd_loss, 
       x_entropy,
       accuracy, 
       tf.argmax(softmax, 1), 
       tf.argmax(labels, 1)], 
      feed_dict = {
        input_placeholder: compds,
        label_placeholder: labels,
      }
    )
    t2 = time.time()
    TP, TN, FP, FN, SEN, SPE, MCC = pk_model.compute_performance(label_dense, prediction)
    format_str = "%6d %6d %6.3f %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %5.3f %5.3f %s"
    logfile.write(format_str % (g_step, g_step, LV, XLV, 0, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))
    logfile.write('\n')
    print(format_str % (g_step, g_step, LV, XLV, 0, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))  

  logfile.close()


if __name__ == "__main__":
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  evaluate("vegfr2", 10001)



