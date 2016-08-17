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

import pk_model

def train(dataset_dict, neg_dataset, target_list, out_filename, batch_size=128,
          ckpt_dir = "/tmp/train", train_from = 0, max_step = 1000, 
          pretrained_variables = False, keep_prob = 0.8,
          start_learning_rate = 0.0, decay_step = 16000, decay_rate = 0.7):

  """ train the model """  
  # train log file
  outfile = open(out_filename, 'w')
  outfile.write("train starts at: %s\n" % datetime.datetime.now())

  with tf.Graph().as_default():
    
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 2048))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))

    global_step = tf.Variable(train_from, trainable=False)

    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step, decay_rate)

    # build a Graph that computes the softmax predictions from the
    # inference model.
    base = pk_model.term(input_placeholder)

    softmax_dict = dict()
    loss_dict = dict()
    accuracy_dict = dict()
    train_op_dict = dict()

    for target in target_list:
      # compute softmax
      softmax_dict[target] = pk_model.branch(target, base)
      # compute loss.
      term_wd_loss = tf.get_collection("term_wd_loss")
      branch_wd_loss = tf.get_collection(target+"_wd_loss")
      x_entropy = pk_model.x_entropy(softmax_dict[target], label_placeholder, target)
      loss_dict[target] = tf.add_n(term_wd_loss + branch_wd_loss + [x_entropy])
      # compute accuracy
      accuracy_dict[target] = pk_model.accuracy(softmax_dict[target], label_placeholder, target)
      # train op
      train_op_dict[target] = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_dict[target], global_step=global_step)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

    # start running operations on the Graph.
    sess = tf.Session()
    
    # initialize all variables at first.
    sess.run(tf.initialize_all_variables())
    """
    # restore from some train steps
    if train_from > 0:
      saver.restore(sess, ckpt_dir + '/model.ckpt-' + str(train_from))
      sess.run(tf.initialize_variables([global_step], name="init_non_trainable_variables"))

    if pretrained_variables:
      # restore from pretrained variables
      variables_to_restore = tf.get_collection("drug_1501")
      saver_pre = tf.train.Saver(variables_to_restore)
      saver_pre.restore(sess, pretrained_variables)
    """

    perm = range(batch_size * 2)

    print("  step g_step   loss learn_rate    TP    FN    TN    FP    SEN    SPE    ACC    MCC sec/step target")
    # train with max step
    for step in xrange(0, max_step):
      neg_compds_batch, neg_labels_batch = neg_dataset.generate_batch(batch_size)

      for target in target_list:
        start_time = time.time()
        pos_compds_batch, pos_labels_batch = dataset_dict[target].generate_batch(batch_size)
        random.shuffle(perm)
        compds_batch = numpy.vstack([pos_compds_batch, neg_compds_batch])[perm]
        labels_batch = numpy.vstack([pos_labels_batch, neg_labels_batch])[perm]
        _, LV, LR, ACC, prediction, label_dense = sess.run(
          [train_op_dict[target], 
           loss_dict[target], 
           learning_rate, 
           accuracy_dict[target], 
           tf.argmax(softmax_dict[target], 1), 
           tf.argmax(labels_batch, 1)], 
          feed_dict = {
            input_placeholder: compds_batch,
            label_placeholder: labels_batch,
          }
        )
        duration = float(time.time() - start_time)
      
        # compute performance
        if step % 100 ==0 or (step + 1) == max_step:
          TP,TN,FP,FN,SEN,SPE,MCC = pk_model.compute_performance(label_dense, prediction)
        
          # print to file and screen
          format_str = "%6d %6d %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %8.3f %s"
          outfile.write(format_str % (step, global_step.eval(sess), LV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, duration, target))
          outfile.write('\n')
          print(format_str % (step, global_step.eval(sess), LV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, duration, target))      

        # save the model checkpoint periodically.
        if step % 30 == 0 or (step + 1) == max_step:
          checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)

  outfile.write("train ends at: %s\n" % datetime.datetime.now())
  outfile.close()

