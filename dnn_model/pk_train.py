#!/usr/bin/python
# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
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
import pk_model


def train(target_list, train_from = 0):

  # dataset
  d = pki.Datasets(target_list)


  # model parameters
  pos_batch_size = 256
  neg_batch_size = pos_batch_size * 10 # the mean number of neg sample is 25.23 times as many as pos's.
  ckpt_dir = "ckpt_files/%d_%d" % (pos_batch_size, neg_batch_size)
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  log_dir = "log_files"
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  step_per_epoch = d.neg.size / pos_batch_size
  max_step = int(5.0 * step_per_epoch)
  keep_prob = 0.8
  start_learning_rate = 0.05
  decay_step = step_per_epoch * 8
  decay_rate = 0.7

  input_vec_len = 6117

   
  # train log file
  log_path = os.path.join(log_dir, "pk_train_%d_%d.log" % (pos_batch_size, neg_batch_size))
  logfile = open(log_path, 'w')
  logfile.write("train starts at: %s\n" % datetime.datetime.now())


  # train the model 
  with tf.Graph().as_default(), tf.device("/gpu:2"):

    # exponential decay learning rate
    global_step = tf.Variable(train_from, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step, decay_rate)

    # build the model
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # build the "Tree" with a mutual "Term" and several "Branches"
    base = pk_model.term(input_placeholder)
    softmax_dict = dict()
    wd_loss_dict = dict()
    x_entropy_dict = dict()
    loss_dict = dict()
    accuracy_dict = dict()
    train_op_dict = dict()
    for target in target_list:
      # compute softmax
      softmax_dict[target] = pk_model.branch(target, base)
      # compute loss.
      wd_loss_dict[target] = tf.add_n(tf.get_collection("term_wd_loss") + tf.get_collection(target+"_wd_loss"))
      x_entropy_dict[target] = pk_model.x_entropy(softmax_dict[target], label_placeholder, target)
      loss_dict[target]  = tf.add(wd_loss_dict[target], x_entropy_dict[target])
      # compute accuracy
      accuracy_dict[target] = pk_model.accuracy(softmax_dict[target], label_placeholder, target)
      # train op
      train_op_dict[target] = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_dict[target], global_step=global_step)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

    # start running operations on the Graph.
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    
    # initialize all variables at first.
    sess.run(tf.initialize_all_variables())
    if train_from != 0:
      saver.restore(sess, "ckpt_files/256_256/pk.ckpt-" + str(train_from))

    # train with max step
    print("  step g_step wdloss x_loss learn_rate    TP    FN    TN    FP    SEN    SPE    ACC    MCC t1-t0 t2-t1 t3-t2  target")
    for step in xrange(max_step):
      for target in target_list:
        t0 = time.time()
        compds_batch, labels_batch = d.next_train_batch(target, pos_batch_size, neg_batch_size)

        t1 = float(time.time())
        _ = sess.run(train_op_dict[target], feed_dict={input_placeholder: compds_batch, label_placeholder: labels_batch})

        t2 = float(time.time())
        # compute performance
        if step % 100 ==0 or (step + 1) == max_step:
          LV, XLV, LR, ACC, prediction, label_dense = sess.run(
            [wd_loss_dict[target], 
             x_entropy_dict[target],
             learning_rate, 
             accuracy_dict[target], 
             tf.argmax(softmax_dict[target], 1), 
             tf.argmax(labels_batch, 1)], 
            feed_dict = {
              input_placeholder: compds_batch,
              label_placeholder: labels_batch,
            }
          )

          TP, TN, FP, FN, SEN, SPE, MCC = pk_model.compute_performance(label_dense, prediction)
        
          t3 = float(time.time())  

          # print to file and screen
          format_str = "%6d %6d %6.3f %6.3f %10.9f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %5.3f %5.3f %5.3f %s"
          logfile.write(format_str % (step, global_step.eval(sess), LV, XLV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, t3-t2, target))
          logfile.write('\n')
          print(format_str % (step, global_step.eval(sess), LV, XLV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, t3-t2, target))      

        # save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_step:
          checkpoint_path = os.path.join(ckpt_dir, 'pk.ckpt')
          saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)



    for target in target_list:
      # the whole train
      compds_batch = numpy.vstack([d.pos[target].features[d.pos[target].train_perm], d.neg.features[d.neg.train_perm]])
      labels_batch = numpy.vstack([d.pos[target].labels[d.pos[target].train_perm], d.neg.mask_dict[target][d.neg.train_perm]])
      LV, XLV, LR, ACC, prediction, label_dense = sess.run(
        [wd_loss_dict[target], 
         x_entropy_dict[target],
         learning_rate, 
         accuracy_dict[target], 
         tf.argmax(softmax_dict[target], 1), 
         tf.argmax(labels_batch, 1)], 
        feed_dict = {
          input_placeholder: compds_batch, 
          label_placeholder: labels_batch, 
        }
      )
  
      TP, TN, FP, FN, SEN, SPE, MCC = pk_model.compute_performance(label_dense, prediction)

      # print to file and screen
      format_str = "%6d %6d %6.3f %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %5.3f %5.3f %s"
      logfile.write(format_str % (step, global_step.eval(sess), LV, XLV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))
      logfile.write('\n')
      print(format_str % (step, global_step.eval(sess), LV, XLV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))    

    for target in target_list:
      # the whole test
      compds_batch = numpy.vstack([d.pos[target].features[d.pos[target].test_perm], d.neg.features[d.neg.test_perm]])
      labels_batch = numpy.vstack([d.pos[target].labels[d.pos[target].test_perm], d.neg.mask_dict[target][d.neg.test_perm]])
      LV, XLV, LR, ACC, prediction, label_dense = sess.run(
        [wd_loss_dict[target], 
         x_entropy_dict[target],
         learning_rate, 
         accuracy_dict[target], 
         tf.argmax(softmax_dict[target], 1), 
         tf.argmax(labels_batch, 1)], 
        feed_dict = {
          input_placeholder: compds_batch,
          label_placeholder: labels_batch, 
        }
      )
  
      TP, TN, FP, FN, SEN, SPE, MCC = pk_model.compute_performance(label_dense, prediction)

      # print to file and screen
      format_str = "%6d %6d %6.3f %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %5.3f %5.3f %s"
      logfile.write(format_str % (step, global_step.eval(sess), LV, XLV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))
      logfile.write('\n')
      print(format_str % (step, global_step.eval(sess), LV, XLV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))    


  logfile.write("train ends at: %s\n" % datetime.datetime.now())
  logfile.close()



if __name__ == "__main__":

  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  train(target_list, train_from=11824)



