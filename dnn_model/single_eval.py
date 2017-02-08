# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Oct 2016
# Addr: Shenzhen, China
# Description: evaluate pk model for a single target

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import datetime
import tensorflow as tf

import dnn_model 
sys.path.append("/home/scw4750/Documents/chembl/data_files/")
import chembl_input as ci


def evaluate(target, g_step_list=None):
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
  # eval log file
  log_dir = "log_files"
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  log_path = os.path.join(log_dir, "eval_%s_%d_%4.3f_%4.3e.log" % (target, batch_size, keep_prob, wd))
  logfile = open(log_path, 'w')
  logfile.write("eval starts at: %s\n" % datetime.datetime.now())

  
  # g_step_list
  #step_list = range(0, 24991, 10 * step_per_epoch)
  g_step_list = range(1, 2235900, 10 * step_per_epoch)
  g_step_list.append(2235900)

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

    # format str
    format_str = "%6d %6.4f %7.5f %10.8f %5d %5d %5d %5d %6.4f %6.4f %6.4f %6.4f %5.3f %5.3f %5.3f %10s "

    # pns
    pns_compds = d.target_pns_features.toarray()
    pns_labels_dense = d.target_pns_mask.values.astype(int)
    pns_labels_one_hot = ci.dense_to_one_hot(pns_labels_dense)

    # target test
    test_compds = d.test_features.toarray()
    test_labels_dense = d.test_labels
    test_labels_one_hot = d.test_labels_one_hot

    # target train
    time_split_train = d.target_clf_label[d.target_clf_label["YEAR"] <= 2014]
    m = d.target_cns_mask.index.isin(time_split_test["CMPD_CHEMBLID"])
    target_train_features = d.target_cns_features[m]
    target_train_labels_dense = d.target_cns_mask[m].values.astype(int)
    target_train_labels_one_hot = ci.dense_to_one_hot(target_train_labels_dense)

    for g_step in g_step_list:
      # Restores variables from checkpoint
      saver.restore(sess, ckpt_path + "-%d" % g_step)

      # the whole pns
      t2 = time.time()
      wd_ls, x_ls, pred = sess.run([wd_loss, x_entropy, tf.argmax(softmax, 1)], 
        feed_dict = {input_placeholder: pns_compds, label_placeholder: pns_labels_one_hot})
      tp, tn, fp, fn, sen, spe, acc, mcc = ci.compute_performance(pns_labels_dense, pred)
      t3 = time.time()
      logfile.write(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, t3-t2, target) + "\n")
      print(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, t3-t2, target))      

      # the whole cns
      t2 = time.time()
      # 878721 / 10000 = 87.8721 < 88
      wd_ls = x_ls = acc = 0
      pred_list = []
      label_dense_list = []
      d.reset_begin_end_cns()
      for i in range(88):
        compds_train, labels_train = d.generate_cns_batch_once(10000)
        x_ls_b, pred_b = sess.run([x_entropy, tf.argmax(softmax, 1)], 
          feed_dict = {input_placeholder: compds_train, label_placeholder: labels_train})
        x_ls += x_ls_b * compds_train.shape[0]
        pred_list.append(pred_b) 
      x_ls /= d.target_cns_features.shape[0]
      pred = np.hstack(pred_list)
      tp, tn, fp, fn, sen, spe, acc, mcc = ci.compute_performance(d.target_cns_mask.values.astype(int), pred)
      t3 = time.time()
      logfile.write(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, t3-t2, target) + "\n")
      print(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, t3-t2, target)) 
    
      # the target's train
      t2 = time.time()
      x_ls, pred= sess.run([x_entropy, tf.argmax(softmax, 1)], 
        feed_dict = {input_placeholder: target_train_features, label_placeholder: target_train_labels_one_hot})
      tp, tn, fp, fn, sen, spe, acc, mcc = ci.compute_performance(target_train_labels_dense, pred)
      t3 = time.time()
      logfile.write(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, t3-t2, target) + "\n")
      print(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, t3-t2, target))    

      # the target's test
      t2 = time.time()
      x_ls, pred= sess.run([x_entropy, tf.argmax(softmax, 1)], 
        feed_dict = {input_placeholder: test_compds, label_placeholder: test_labels_one_hot})
      tp, tn, fp, fn, sen, spe, acc, mcc = ci.compute_performance(test_labels_dense, pred)
      t3 = time.time()
      logfile.write(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, t3-t2, target) + "\n")
      print(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, t3-t2, target))    


  logfile.write("eval ends at: %s\n" % datetime.datetime.now())
  logfile.close()


if __name__ == "__main__":
  # the newly picked out 15 targets, include 9 targets from 5 big group, and 6 targets from others.
  target_list = ["CHEMBL279", "CHEMBL203", # Protein Kinases
               "CHEMBL217", "CHEMBL253", # GPCRs (Family A)
               "CHEMBL235", "CHEMBL206", # Nuclear Hormone Receptors
               "CHEMBL240", "CHEMBL4296", # Voltage Gated Ion Channels
               "CHEMBL4805", # Ligand Gated Ion Channels
               "CHEMBL204", "CHEMBL244", "CHEMBL4822", "CHEMBL340", "CHEMBL205", "CHEMBL4005" # Others
              ] 

  #for target in target_list:
  target = "CHEMBL203"
  evaluate(target)



