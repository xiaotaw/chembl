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



def train(target, batch_size=256, train_from = 0):

  """ train the model 
  """
  log_dir = "log_files"
  ckpt_dir = os.path.join("ckpt_files", target)
  max_step = 20000
  keep_prob = 0.8,
  start_learning_rate = 0.05
  decay_step = 5000
  decay_rate = 0.7
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  pos_dataset = pk_input.get_inputs_by_cpickle("data_files/pkl_files/" + target + "_train.pkl")
  neg_dataset = pk_input.get_inputs_by_cpickle("data_files/pkl_files/pubchem_neg_sample.pkl")

  # train log file
  log_path = os.path.join(log_dir, target + "_train_single.log")
  logfile = open(log_path, 'w')
  logfile.write("train starts at: %s\n" % datetime.datetime.now())

  with tf.Graph().as_default():
    
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 2048))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))

    global_step = tf.Variable(train_from, trainable=False)

    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_step, decay_rate)

    # build a Graph that computes the softmax predictions from the
    # inference model.
    base = pk_model.term(input_placeholder)


    # compute softmax
    softmax = pk_model.branch(target, base)
    # compute loss.
    term_wd_loss = tf.get_collection("term_wd_loss")
    branch_wd_loss = tf.get_collection(target+"_wd_loss")
    x_entropy = pk_model.x_entropy(softmax, label_placeholder, target)
    loss  = tf.add_n(term_wd_loss + branch_wd_loss + [x_entropy])
    # compute accuracy
    accuracy = pk_model.accuracy(softmax, label_placeholder, target)
    # train op
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

    # start running operations on the Graph.
    sess = tf.Session()
    
    # initialize all variables at first.
    sess.run(tf.initialize_all_variables())

    perm = range(batch_size)
    compds_batch = numpy.zeros([batch_size, 2048])
    labels_batch = numpy.zeros([batch_size, 2])

    print("  step g_step   loss learn_rate    TP    FN    TN    FP    SEN    SPE    ACC    MCC sec/step target")
    # train with max step
    for step in xrange(0, max_step):

      # prepare batch data
      start_time = time.time()
      compds_batch[:batch_size//2], labels_batch[:batch_size//2] = neg_dataset.generate_batch(batch_size//2)
      compds_batch[batch_size//2:], labels_batch[batch_size//2:] = pos_dataset.generate_batch(batch_size//2)
      random.shuffle(perm)
      numpy.clip(compds_batch, 0, 1, out=compds_batch)
      numpy.clip(labels_batch, 0, 1, out=labels_batch)
      compds_batch = compds_batch[perm]
      labels_batch = labels_batch[perm]

      _, LV, LR, ACC, prediction, label_dense = sess.run(
        [train_op, 
         loss, 
         learning_rate, 
         accuracy, 
         tf.argmax(softmax, 1), 
         tf.argmax(labels_batch, 1)], 
        feed_dict = {
          input_placeholder: compds_batch,
          label_placeholder: labels_batch,
        }
      )
      duration = float(time.time() - start_time)
      
      # compute performance
      if step % 100 == 0 or (step + 1) == max_step:
        TP,TN,FP,FN,SEN,SPE,MCC = pk_model.compute_performance(label_dense, prediction)
        
        # print to file and screen
        format_str = "%6d %6d %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %8.3f %s"
        logfile.write(format_str % (step, global_step.eval(sess), LV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, duration, target))
        logfile.write('\n')
        print(format_str % (step, global_step.eval(sess), LV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, duration, target))      


      if step % 1000 == 0 or (step + 1) == max_step:
                


        LV, LR, ACC, prediction, label_dense = sess.run(
          [loss, 
           learning_rate, 
           accuracy, 
           tf.argmax(softmax, 1), 
           tf.argmax(numpy.vstack([pos_dataset.labels, neg_dataset.labels]), 1)], 
          feed_dict = {
            input_placeholder: numpy.vstack([pos_dataset.compds, neg_dataset.compds]),
            label_placeholder: numpy.vstack([pos_dataset.labels, neg_dataset.labels]),
          }
        )
        
        TP,TN,FP,FN,SEN,SPE,MCC = pk_model.compute_performance(label_dense, prediction)
        # print to file and screen
        format_str = "%6d %6d %6.3f %10.3f %5d %5d %5d %5d %6.3f %6.3f %6.3f %6.3f %8.3f %s"
        logfile.write(format_str % (step, global_step.eval(sess), LV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, duration, "test"))
        logfile.write('\n')
        print(format_str % (step, global_step.eval(sess), LV, LR, TP, FN, TN, FP, SEN, SPE, ACC, MCC, duration, "test"))      

      # save the model checkpoint periodically.
      #if step % 500 == 0 or (step + 1) == max_step:
        #checkpoint_path = os.path.join(ckpt_dir, target, 'model.ckpt')
        #saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)

  logfile.write("train ends at: %s\n" % datetime.datetime.now())
  logfile.close()

if __name__ == "__main__":
  train("cdk2")

