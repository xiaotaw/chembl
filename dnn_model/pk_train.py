# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Nov 2016
# Addr: Shenzhen, China
# Description: train pk model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import datetime
import math
import numpy
import random
import tensorflow as tf

import pk_input as pki
import dnn_model


def train(target_list, train_from = 0):

  # dataset
  d = pki.Datasets(target_list)

  # batch size.
  # note: the mean number of neg sample is 25.23 times as many as pos's.
  neg_batch_size = 512
  pos_batch_size_dict = {}
  pos_sum = 0
  for target in target_list:
    pos_sum += d.pos[target].size
  pos_batch_size = int(neg_batch_size * pos_sum / d.neg.size)
  for target in target_list:
    pos_batch_size_dict[target] = int(neg_batch_size * d.pos[target].size / d.neg.size)
    #pos_batch_size_dict[target] = pos_batch_size
  # learning rate 
  step_per_epoch = int(d.neg.size / neg_batch_size)
  start_learning_rate = 0.05
  decay_step = step_per_epoch * 10 * 8
  decay_rate = 0.9 
  # max train steps
  max_step = 50 * step_per_epoch
  # input vec_len
  input_vec_len = d.neg.features.shape[1]
  # keep prob
  keep_prob = 0.8
  # weight decay
  wd = 0.001
  # checkpoint file
  ckpt_dir = "ckpt_files_big_tree/pk"
  ckpt_path = os.path.join(ckpt_dir, '%d_%4.3f_%4.3e.ckpt' % (neg_batch_size, keep_prob, wd))
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  # train log file
  log_dir = "log_files_big_tree"
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  log_path = os.path.join(log_dir, "train_pk_%d_%4.3f_%4.3e.log" % (neg_batch_size, keep_prob, wd))
  logfile = open(log_path, 'w')
  logfile.write("train starts at: %s\n" % datetime.datetime.now())
   

  # train the model 
  with tf.Graph().as_default(), tf.device("/gpu:0"):

    # exponential decay learning rate
    global_step = tf.Variable(train_from, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step, decay_rate)

    # build the model
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # build the "Tree" with a mutual "Term" and several "Branches"
    base = dnn_model.term(input_placeholder, wd=wd, keep_prob=keep_prob)
    softmax_dict = dict()
    wd_loss_dict = dict()
    x_entropy_dict = dict()
    loss_dict = dict()
    accuracy_dict = dict()
    train_op_dict = dict()
    for target in target_list:
      # compute softmax
      softmax_dict[target] = dnn_model.branch(target, base, wd=wd, keep_prob=keep_prob)
      # compute loss.
      wd_loss_dict[target] = tf.add_n(tf.get_collection("term_wd_loss") + tf.get_collection(target+"_wd_loss"))
      x_entropy_dict[target] = dnn_model.x_entropy(softmax_dict[target], label_placeholder, target)
      loss_dict[target]  = tf.add(wd_loss_dict[target], x_entropy_dict[target])
      # compute accuracy
      accuracy_dict[target] = dnn_model.accuracy(softmax_dict[target], label_placeholder, target)
      # train op
      train_op_dict[target] = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_dict[target], global_step=global_step)
    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)
    # start running operations on the Graph.
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
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

    # train with max step
    for step in xrange(max_step):
      for target in target_list:
        t0 = time.time()

        # get a batch sample
        compds_batch, labels_batch = d.next_train_batch(target, pos_batch_size_dict[target], neg_batch_size)
        t1 = float(time.time())

        _ = sess.run(train_op_dict[target], feed_dict={input_placeholder: compds_batch, label_placeholder: labels_batch})
        t2 = float(time.time())

        # compute performance
        # compute performance
        if step % step_per_epoch == 0 or (step + 1) == max_step:
          g_step, wd_ls, x_ls, lr, acc, pred, label_dense = sess.run([global_step, wd_loss_dict[target], x_entropy_dict[target], learning_rate, accuracy_dict[target], tf.argmax(softmax_dict[target], 1), tf.argmax(labels_batch, 1)],
            feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch})
          tp, tn, fp, fn, sen, spe, mcc = dnn_model.compute_performance(label_dense, pred)
          t3 = float(time.time())       
          # print to file and screen

          logfile.write(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))
          logfile.write('\n')
          print(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))      


      # save the model checkpoint periodically.
      if step % (10 * step_per_epoch) == 0 or (step + 1) == max_step:
        saver.save(sess, ckpt_path, global_step=global_step, write_meta_graph=False)

      if (step > 3 * 10 * step_per_epoch) and (step % (10 * step_per_epoch) == 0 or (step + 1) == max_step):
        for target in target_list:
          # the whole train
          t0 = time.time()
          compds_batch = numpy.vstack([d.pos[target].features[d.pos[target].train_perm], d.neg.features[d.neg.train_perm]])
          labels_batch = numpy.vstack([d.pos[target].labels[d.pos[target].train_perm], d.neg.mask_dict[target][d.neg.train_perm]])
          t1 = time.time()
          t2 = time.time()
          g_step, wd_ls, x_ls, lr, acc, pred, label_dense = sess.run([global_step, wd_loss_dict[target], x_entropy_dict[target], learning_rate, accuracy_dict[target], tf.argmax(softmax_dict[target], 1), tf.argmax(labels_batch, 1)],
            feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch})
          t3 = float(time.time()) 
          tp, tn, fp, fn, sen, spe, mcc = dnn_model.compute_performance(label_dense, pred)
          # print to file and screen
          logfile.write(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))
          logfile.write('\n')
          print(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))    

          # the whole test
          t0 = time.time()
          compds_batch = numpy.vstack([d.pos[target].features[d.pos[target].test_perm], d.neg.features[d.neg.test_perm]])
          labels_batch = numpy.vstack([d.pos[target].labels[d.pos[target].test_perm], d.neg.mask_dict[target][d.neg.test_perm]])
          t1 = time.time()
          t2 = time.time()
          g_step, wd_ls, x_ls, lr, acc, pred, label_dense = sess.run([global_step, wd_loss_dict[target], x_entropy_dict[target], learning_rate, accuracy_dict[target], tf.argmax(softmax_dict[target], 1), tf.argmax(labels_batch, 1)],
            feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch})
          t3 = float(time.time()) 
          tp, tn, fp, fn, sen, spe, mcc = dnn_model.compute_performance(label_dense, pred)
          # print to file and screen
          logfile.write(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))
          logfile.write('\n')
          print(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target)) 


  logfile.write("train ends at: %s\n" % datetime.datetime.now())
  logfile.close()



if __name__ == "__main__":

  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  train(target_list, train_from=0)



