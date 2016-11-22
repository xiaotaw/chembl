# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time Created: Aug 2016
# Time Last Updated: Oct 2016
# Addr: Shenzhen, China
# Description: evaluate pk model for a single target

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy
import datetime
import tensorflow as tf

import pk_model 
import pk_input as pki


def evaluate(d, target, g_step_list=None):
  """ evaluate the model 
  """
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
  max_step = 700 * step_per_epoch
  # input vec_len
  input_vec_len = d.neg.features.shape[1]
  # keep prob
  keep_prob = 0.8
  # weight decay
  wd = 0.004
  # checkpoint file
  ckpt_dir = "ckpt_files/%s" % target
  ckpt_path = os.path.join(ckpt_dir, '%d_%d_%4.3f_%4.3e.ckpt' % (pos_batch_size, neg_batch_size, keep_prob, wd))
  #ckpt_path = os.path.join(ckpt_dir, '%d_%d.ckpt' % (pos_batch_size, neg_batch_size))
  # eval log file
  log_dir = "log_files"
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  log_path = os.path.join(log_dir, "eval_%s_%d_%d_%4.3f_%4.3e.log" % (target, pos_batch_size, neg_batch_size, keep_prob, wd))
  logfile = open(log_path, 'w')
  logfile.write("eval starts at: %s\n" % datetime.datetime.now())

  
  # g_step_list
  #step_list = range(0, 24991, 10 * step_per_epoch)
  g_step_list = range(30871, 44100, 10 * step_per_epoch)
  g_step_list.append(44100)

  with tf.Graph().as_default(), tf.device("/gpu:1"):
    
    # build the model
    input_placeholder = tf.placeholder(tf.float32, shape = (None, input_vec_len))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, 2))
    # build the "Tree" with a mutual "Term" and several "Branches"
    base = pk_model.term(input_placeholder, wd=wd, keep_prob=1.0)
    # compute softmax
    softmax = pk_model.branch(target, base, wd=wd, keep_prob=1.0)
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(config=config)

    # format str
    format_str = "%6d %6.4f %7.5f %10.8f %5d %5d %5d %5d %6.4f %6.4f %6.4f %6.4f %5.3f %5.3f %5.3f %10s "


    compds_train = numpy.vstack([d.pos[target].features[d.pos[target].train_perm], d.neg.features[d.neg.train_perm]])
    labels_train = numpy.vstack([d.pos[target].labels[d.pos[target].train_perm], d.neg.mask_dict[target][d.neg.train_perm]])


    compds_test = numpy.vstack([d.pos[target].features[d.pos[target].test_perm], d.neg.features[d.neg.test_perm]])
    labels_test = numpy.vstack([d.pos[target].labels[d.pos[target].test_perm], d.neg.mask_dict[target][d.neg.test_perm]])

    for g_step in g_step_list:
      # Restores variables from checkpoint
      saver.restore(sess, ckpt_path + "-%d" % g_step)

      # the whole train
      t0 = time.time()
      t1 = time.time()
      t2 = time.time()
      wd_ls, x_ls, acc, pred, label_dense = sess.run([wd_loss, x_entropy, accuracy, tf.argmax(softmax, 1), tf.argmax(labels_train, 1)], 
        feed_dict = {input_placeholder: compds_train, label_placeholder: labels_train})
      tp, tn, fp, fn, sen, spe, mcc = pk_model.compute_performance(label_dense, pred)
      t3 = time.time()
      logfile.write(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))
      print(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))    

      # the whole test
      t0 = time.time()
      t1 = time.time()
      t2 = time.time()
      wd_ls, x_ls, acc, pred, label_dense = sess.run([wd_loss, x_entropy, accuracy, tf.argmax(softmax, 1), tf.argmax(labels_test, 1)], 
        feed_dict = {input_placeholder: compds_test, label_placeholder: labels_test})
      tp, tn, fp, fn, sen, spe, mcc = pk_model.compute_performance(label_dense, pred)
      t3 = time.time()
      logfile.write(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))
      logfile.write('\n')
      print(format_str % (g_step, wd_ls, x_ls, 0, tp, fn, tn, fp, sen, spe, acc, mcc, t1-t0, t2-t1, t3-t2, target))    

  logfile.write("eval ends at: %s\n" % datetime.datetime.now())
  logfile.close()


if __name__ == "__main__":
  #target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]
  target_list = ["gsk3b"]

  d = pki.Datasets(target_list)

  for target in target_list:
    evaluate(d, target)



