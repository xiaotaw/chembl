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

import pk_input
import pk_model


def train(target_list, train_from = 0):

  # model parameters
  batch_size = 128
  ckpt_dir = "ckpt_files"
  log_dir = "log_files"
  max_step = 5000
  keep_prob = 0.8
  start_learning_rate = 0.05
  decay_step = 5000
  decay_rate = 0.7


   
  # train log file
  log_path = os.path.join(log_dir, "pk_train.log")
  logfile = open(log_path, 'w')
  logfile.write("train starts at: %s\n" % datetime.datetime.now())


  # get input dataset
  train_dataset_dict = dict()
  test_dataset_dict = dict()
  for target in target_list:
    train_dataset_dict[target] = pk_input.get_inputs_by_cpickle("data_files/pkl_files/" + target + "_train.pkl", clip=True) 
    test_dataset_dict[target] = pk_input.get_inputs_by_cpickle("data_files/pkl_files/" + target + "_test.pkl", clip=True) 

  neg_dataset = pk_input.get_inputs_by_cpickle("data_files/pkl_files/pubchem_neg_sample.pkl", clip=True)



  # train the model 
  with tf.Graph().as_default(), tf.device("/gpu:0"):

    # exponential decay learning rate
    global_step = tf.Variable(train_from, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step, decay_rate)

    # build the model
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 8192))
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

    # define train batch
    perm = range(batch_size * 2)
    compds_batch = numpy.zeros([batch_size * 2, 8192])
    labels_batch = numpy.zeros([batch_size * 2, 2])

    # train with max step
    print("  step g_step wdloss x_loss learn_rate    TP    FN    TN    FP    SEN    SPE    ACC    MCC t1-t0 t2-t1  target")
    for step in xrange(0, max_step):
      for target in target_list:
        t0 = time.time()
        compds_batch[: batch_size], labels_batch[: batch_size] = neg_dataset.generate_batch(batch_size)
        compds_batch[batch_size: ], labels_batch[batch_size: ] = train_dataset_dict[target].generate_batch(batch_size)
        random.shuffle(perm)
        compds_batch = compds_batch[perm]
        labels_batch = labels_batch[perm]

        _ = sess.run(train_op_dict[target], feed_dict={input_placeholder: compds_batch, label_placeholder: labels_batch})

        t1 = float(time.time())

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
        
          t2 = float(time.time())  

          # print to file and screen
          format_str = "%6d %6d %6.3f %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %5.3f %5.3f %s"
          logfile.write(format_str % (step, global_step.eval(sess), LV, XLV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))
          logfile.write('\n')
          print(format_str % (step, global_step.eval(sess), LV, XLV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, t1-t0, t2-t1, target))      

        # save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_step:
          checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)

    
    for target in target_list:
      # the whole train
      compds_batch = numpy.vstack([train_dataset_dict[target].compds, neg_dataset.compds])
      labels_batch = numpy.vstack([train_dataset_dict[target].labels, neg_dataset.labels])
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
      LV, XLV, LR, ACC, prediction, label_dense = sess.run(
        [wd_loss_dict[target], 
         x_entropy_dict[target],
         learning_rate, 
         accuracy_dict[target], 
         tf.argmax(softmax_dict[target], 1), 
         tf.argmax(test_dataset_dict[target].labels, 1)], 
        feed_dict = {
          input_placeholder: test_dataset_dict[target].compds, 
          label_placeholder: test_dataset_dict[target].labels, 
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

  train(target_list)



