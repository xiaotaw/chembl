# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Dec 2016
# Addr: Shenzhen, China
# Description: train chembl model for a single target

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import getpass
import datetime
import tensorflow as tf

sys.path.append("/home/%s/Documents/chembl/data_files/" % getpass.getuser())
import dnn_model
import chembl_input as ci


def train(target, train_from = 0):
  """"""
  # dataset
  d = ci.Dataset(target, train_pos_multiply=2)
  # batch size
  batch_size = 128
  # learning rate 
  step_per_epoch = int(d.train_size / batch_size)
  start_learning_rate = 0.05
  decay_step = step_per_epoch * 10
  decay_rate = 0.9
  # max train steps
  max_step = 300 * step_per_epoch
  # input vec_len
  input_vec_len = d.train_features.shape[1]
  # keep prob
  keep_prob = 0.8
  # weight decay
  wd = 0.004
  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%4.3f_%4.3e.ckpt' % (batch_size, keep_prob, wd))
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  # train log file
  log_dir = "log_files"
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  log_path = os.path.join(log_dir, "train_%s_%d_%4.3f_%4.3e.log" % (target, batch_size, keep_prob, wd))
  logfile = open(log_path, 'w')
  logfile.write("train starts at: %s\n" % datetime.datetime.now())


  # build dnn model and train
  with tf.Graph().as_default(), tf.device('/gpu:0'):
    # placeholders
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # global step and learning rate
    global_step = tf.Variable(train_from, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step, decay_rate)
    # build a Graph that computes the softmax predictions from the
    # inference model.
    base = dnn_model.term(input_placeholder, in_units=input_vec_len, wd=wd, keep_prob=keep_prob)
    # compute softmax
    softmax = dnn_model.branch(target, base, wd=wd, keep_prob=keep_prob)
    # compute loss.
    wd_loss = tf.add_n(tf.get_collection("term_wd_loss") + tf.get_collection(target+"_wd_loss"))
    x_entropy = dnn_model.x_entropy(softmax, label_placeholder, target, neg_weight=1)
    loss  = tf.add(wd_loss, x_entropy)
    # train op
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)
    # start running operations on the Graph.
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(config=config)
    # initialize all variables at first.
    sess.run(tf.initialize_all_variables())
    if train_from != 0:
      saver.restore(sess, ckpt_path + "-%d" % train_from)
    # print title to screen and log file
    title_str = "  step g_step wdloss   xloss learn_rate    TP    FN    TN    FP    SEN    SPE    ACC    MCC t1-t0 t2-t1 t3-t2  target"
    print(title_str)
    logfile.write(title_str + "\n")

    # format str
    format_str = "%6d %6d %6.4f %7.5f %10.8f %5d %5d %5d %5d %6.4f %6.4f %6.4f %6.4f %5.3f %5.3f %5.3f %10s "

    # train the model
    for step in xrange(max_step):
      t0 = time.time()

      # get a batch sample
      perm = d.generate_perm_for_train_batch(batch_size)
      compds_batch = d.train_features[perm].toarray()
      labels_batch_one_hot = d.train_labels_one_hot[perm]
      labels_batch_dense = d.train_labels[perm]
      t1 = time.time()

      # train once
      _ = sess.run([train_op],feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch_one_hot})
      t2 = time.time()

      # compute performance for the train batch
      if step % step_per_epoch == 0 or (step + 1) == max_step:
        g_step, wd_ls, x_ls, lr, pred = sess.run([global_step, wd_loss, x_entropy, learning_rate, tf.argmax(softmax, 1)],
          feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch_one_hot})
        tp, tn, fp, fn, sen, spe, acc, mcc = ci.compute_performance(labels_batch_dense, pred)
        t3 = float(time.time())    
        logfile.write(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target) + "\n")
        print(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))      

      # save the model checkpoint periodically.
      if step % (10 * step_per_epoch) == 0 or (step + 1) == max_step:
        saver.save(sess, ckpt_path, global_step=global_step, write_meta_graph=False)

      # compute performance for the test data
      if step % (10 * step_per_epoch) == 0 or (step + 1) == max_step:
        test_compds_batch = d.test_features_dense
        test_labels_batch_one_hot = d.test_labels_one_hot
        test_labels_batch_dense = d.test_labels
        x_ls, pred = sess.run([x_entropy, tf.argmax(softmax, 1)],
          feed_dict = {input_placeholder: test_compds_batch, label_placeholder: test_labels_batch_one_hot})
        tp, tn, fp, fn, sen, spe, acc, mcc = ci.compute_performance(test_labels_batch_dense, pred)
        logfile.write(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, 0, target) + "\n")
        print(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, 0, 0, 0, target)) 

  logfile.write("train ends at: %s\n" % datetime.datetime.now())
  logfile.close()

if __name__ == "__main__":

  # the newly picked out 15 targets, include 9 targets from 5 big group, and 6 targets from others.
  target_list = ["CHEMBL279", 
               "CHEMBL4805", # Ligand Gated Ion Channels
               "CHEMBL244", "CHEMBL4822", "CHEMBL340", "CHEMBL205", "CHEMBL4005" # Others
              ] 


  #for target in target_list:
  target = "CHEMBL4805"
  train(target, train_from=0)

