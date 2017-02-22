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
import datetime
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

import dnn_model 
sys.path.append("/home/scw4750/Documents/chembl/data_files/")
import chembl_input as ci


eval_batch_size = 1024


def evaluate(target, g_step_list=None, gpu_num=0, 
             keep_prob=0.8, wd=0.004, batch_size=128):
  """ evaluate the model 
  """
  # dataset
  d = ci.Dataset(target)
  # learning rate 
  step_per_epoch = int(d.train_size / batch_size)
  # input vec_len
  input_vec_len = d.num_features  
  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%4.3f_%4.3e.ckpt' % (batch_size, keep_prob, wd))

  # pred file
  pred_dir = "pred_files/%s" % target
  if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

  print("%s eval starts at: %s\n" % (target, datetime.datetime.now()))
  
  # g_step_list
  #g_step_list = range(1, 2235900, 10 * step_per_epoch)
  #g_step_list.append(2235900)

  with tf.Graph().as_default(), tf.device("/gpu: %d" % gpu_num):
    # build the model
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # build the "Tree" with a mutual "Term" and several "Branches"
    base = dnn_model.term(input_placeholder, in_units=input_vec_len, wd=wd, keep_prob=1.0)
    # compute softmax
    softmax = dnn_model.branch(target, base, wd=wd, keep_prob=1.0)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())
    # create session.
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)


    for g_step in g_step_list:
      # Restores variables from checkpoint
      saver.restore(sess, ckpt_path + "-%d" % g_step)

      # the whole pns
      pns_pred_file = open(pred_dir + "/pns_%s_%d_%4.3f_%4.3e_%d.pred" % (target, batch_size, keep_prob, wd, g_step), "w") 
      for ids, features, mask in d.batch_generator_pns(eval_batch_size):
        sm = sess.run(softmax, feed_dict={input_placeholder: features})
        for i, s, m in zip(ids, sm[:, 1], mask):
          pns_pred_file.write("%s\t%f\t%d\n" % (i, s, m))
      pns_pred_file.close()

      # the whole cns
      cns_pred_file = open(pred_dir + "/cns_%s_%d_%4.3f_%4.3e_%d.pred" % (target, batch_size, keep_prob, wd, g_step), "w") 
      for ids, features, mask in d.batch_generator_cns(eval_batch_size):
        sm = sess.run(softmax, feed_dict={input_placeholder: features})
        for i, s, m in zip(ids, sm[:, 1], mask):
          cns_pred_file.write("%s\t%f\t%d\n" % (i, s, m))
      cns_pred_file.close()   

      # the target's train
      train_pred_file = open(pred_dir + "/train_%s_%d_%4.3f_%4.3e_%d.pred" % (target, batch_size, keep_prob, wd, g_step), "w")
      sm = sess.run(softmax, feed_dict={input_placeholder: d.target_features_train.toarray()})
      for i, s, m in zip(d.target_ids_train, sm[:, 1], d.target_labels_train):
        train_pred_file.write("%s\t%f\t%d\n" % (i, s, m))
      train_pred_file.close()     

      # the target's test
      test_pred_file = open(pred_dir + "/test_%s_%d_%4.3f_%4.3e_%d.pred" % (target, batch_size, keep_prob, wd, g_step), "w")
      sm = sess.run(softmax, feed_dict={input_placeholder: d.target_features_test.toarray()})
      for i, s, m in zip(d.target_ids_test, sm[:, 1], d.target_labels_test):
        test_pred_file.write("%s\t%f\t%d\n" % (i, s, m))
      test_pred_file.close()   

  print("eval ends at: %s\n" % datetime.datetime.now())


def test(target, g_step):
  # dataset
  d = ci.DatasetTarget(target)  
  # batch size
  batch_size = 128
  # keep prob
  keep_prob = 0.8
  # weight decay
  wd = 0.004
  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%4.3f_%4.3e.ckpt' % (batch_size, keep_prob, wd))
  # input vec_len
  input_vec_len = d.num_features

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
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)

    saver.restore(sess, ckpt_path + "-%d" % g_step)
    sm = sess.run(softmax, feed_dict = {input_placeholder: d.target_features_test.toarray()})

    fpr, tpr, _ = roc_curve(d.target_labels_test, sm[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="r", lw=2, label="ROC curve (area = %.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic of DNN model on %s" % target)
    plt.legend(loc="lower right")
    plt.savefig("%s.png" % target)
    #plt.show()
    


if __name__ == "__main__":
  # the newly picked out 15 targets, include 9 targets from 5 big group, and 6 targets from others.
  target_list = ["CHEMBL279", "CHEMBL203", # Protein Kinases
               "CHEMBL217", "CHEMBL253", # GPCRs (Family A)
               "CHEMBL235", "CHEMBL206", # Nuclear Hormone Receptors
               "CHEMBL240", "CHEMBL4296", # Voltage Gated Ion Channels
               "CHEMBL4805", # Ligand Gated Ion Channels
               "CHEMBL204", "CHEMBL244", "CHEMBL4822", "CHEMBL340", "CHEMBL205", "CHEMBL4005" # Others
              ] 

  target_list = ["CHEMBL203", "CHEMBL204", "CHEMBL205",
                 "CHEMBL206", "CHEMBL217", "CHEMBL235", "CHEMBL240",
                 "CHEMBL244", "CHEMBL253", "CHEMBL279", "CHEMBL340", 
                 "CHEMBL4005", "CHEMBL4296", "CHEMBL4805", "CHEMBL4822", 
                ] 

  g_list = [2161371, 2236500, 2235600, 
            2091321, 2161661, 2086841, 2020411,
            2161951, 2012041, 2161661, 2246400, 
            2235900, 2238000, 2168041,  1936221
           ]

  i = int(sys.argv[1])
  target = target_list[i]
  g_step = g_list[i]
  evaluate(target=target, g_step_list=[g_step], gpu_num=i % 4)

  #test(target, g_step, )



