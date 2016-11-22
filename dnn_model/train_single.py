# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Oct 2016
# Addr: Shenzhen, China
# Description: train pk model for a single target

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import numpy
import random
import datetime
import tensorflow as tf

import pk_input as pki
import pk_model


def train(d, target, train_from = 0):
  """"""
  # batch size.
  # note: the mean number of neg sample is 25.23 times as many as pos's.
  neg_batch_size = 512
  pos_batch_size = int(neg_batch_size * d.pos[target].size / d.neg.size) 
  # learning rate 
  step_per_epoch = int(d.neg.size / neg_batch_size)
  start_learning_rate = 0.05
  decay_step = step_per_epoch * 10
  decay_rate = 0.9
  # max train steps
  max_step = 300 * step_per_epoch
  # input vec_len
  input_vec_len = d.neg.features.shape[1]
  # keep prob
  keep_prob = 0.8
  # weight decay
  wd = 0.004
  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%d_%4.3f_%4.3e.ckpt' % (pos_batch_size, neg_batch_size, keep_prob, wd))
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  # train log file
  log_dir = "log_files"
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  log_path = os.path.join(log_dir, "train_%s_%d_%d_%4.3f_%4.3e.log" % (target, pos_batch_size, neg_batch_size, keep_prob, wd))
  logfile = open(log_path, 'w')
  logfile.write("train starts at: %s\n" % datetime.datetime.now())


  # build dnn model and train
  with tf.Graph().as_default(), tf.device('/gpu:3'):
    # placeholders
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # global step and learning rate
    global_step = tf.Variable(train_from, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step, decay_rate)
    # build a Graph that computes the softmax predictions from the
    # inference model.
    base = pk_model.term(input_placeholder, wd=wd, keep_prob=keep_prob)
    # compute softmax
    softmax = pk_model.branch(target, base, wd=wd, keep_prob=keep_prob)
    # compute loss.
    wd_loss = tf.add_n(tf.get_collection("term_wd_loss") + tf.get_collection(target+"_wd_loss"))
    x_entropy = pk_model.x_entropy(softmax, label_placeholder, target, neg_weight=1)
    loss  = tf.add(wd_loss, x_entropy)
    # compute accuracy
    accuracy = pk_model.accuracy(softmax, label_placeholder, target)
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
      compds_batch, labels_batch = d.next_train_batch(target, pos_batch_size, neg_batch_size)
      t1 = time.time()
      # train once
      _ = sess.run([train_op],feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch})
      t2 = time.time()

      # compute performance
      if step % step_per_epoch == 0 or (step + 1) == max_step:
        g_step, wd_ls, x_ls, lr, acc, pred, label_dense = sess.run([global_step, wd_loss, x_entropy, learning_rate, accuracy, tf.argmax(softmax, 1), tf.argmax(labels_batch, 1)],
          feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch})
        tp, tn, fp, fn, sen, spe, mcc = pk_model.compute_performance(label_dense, pred)
        t3 = float(time.time())       
        # print to file and screen

        logfile.write(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))
        logfile.write('\n')
        print(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))      


      # save the model checkpoint periodically.
      if step % (10 * step_per_epoch) == 0 or (step + 1) == max_step:
        saver.save(sess, ckpt_path, global_step=global_step, write_meta_graph=False)

      if step % (10 * step_per_epoch) == 0 or (step + 1) == max_step:
        # the whole train
        t0 = time.time()
        compds_batch = numpy.vstack([d.pos[target].features[d.pos[target].train_perm], d.neg.features[d.neg.train_perm]])
        labels_batch = numpy.vstack([d.pos[target].labels[d.pos[target].train_perm], d.neg.mask_dict[target][d.neg.train_perm]])
        t1 = time.time()
        t2 = time.time()
        g_step, wd_ls, x_ls, lr, acc, pred, label_dense = sess.run([global_step, wd_loss, x_entropy, learning_rate, accuracy, tf.argmax(softmax, 1), tf.argmax(labels_batch, 1)],
          feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch})
        t3 = float(time.time()) 
        tp, tn, fp, fn, sen, spe, mcc = pk_model.compute_performance(label_dense, pred)
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
        g_step, wd_ls, x_ls, lr, acc, pred, label_dense = sess.run([global_step, wd_loss, x_entropy, learning_rate, accuracy, tf.argmax(softmax, 1), tf.argmax(labels_batch, 1)],
          feed_dict = {input_placeholder: compds_batch, label_placeholder: labels_batch})
        t3 = float(time.time()) 
        tp, tn, fp, fn, sen, spe, mcc = pk_model.compute_performance(label_dense, pred)
        # print to file and screen
        logfile.write(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))
        logfile.write('\n')
        print(format_str % (step, g_step, wd_ls, x_ls, lr, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target)) 

  logfile.write("train ends at: %s\n" % datetime.datetime.now())
  logfile.close()

if __name__ == "__main__":

  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  # dataset
  d = pki.Datasets(target_list)

  #for target in target_list:
  target = target_list[2]
  train(d, target, train_from=0)

