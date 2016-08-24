# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: apply pk model to pubchem dataset, to screen potential active substrate(drugs)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import numpy
import cPickle
import datetime
import tensorflow as tf
from scipy import sparse

import pk_model


def virtual_screening(target_list, part_num):

  # virtual screen log file
  log_dir = "log_files"
  logpath = os.path.join(log_dir, "virtual_screen_pubchem_%d.log" % part_num)
  logfile = open(logpath, "w")
  logfile.write("virtual screen %d starts at: %s\n" % (part_num, datetime.datetime.now()))

  # input and output dir
  pkl_dir = "/raid/xiaotaw/pubchem/pkl_files"
  prediction_dir = "/raid/xiaotaw/pubchem/prediction_files"
  if not os.path.exists(prediction_dir):
    os.mkdir(prediction_dir)

  # screening
  with tf.Graph().as_default(), tf.device("/gpu:%d" % (part_num // 3)):
    # the input
    input_placeholder = tf.placeholder(tf.float32, shape = (None, 2048))

    # the term
    base = pk_model.term(input_placeholder, keep_prob=1.0)

    # the branches
    softmax_dict = dict()
    for target in target_list:
      softmax_dict[target] = pk_model.branch(target, base, keep_prob=1.0)

    # create a saver.
    saver = tf.train.Saver(tf.trainable_variables())

    # Start screen
    prediction_dict = dict()
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    with tf.Session(config=config) as sess:
      # Restores variables from checkpoint
      saver.restore(sess, "ckpt_files/model.ckpt-40000")

      
      #for i in xrange(1, 121225001, 25000):
      begin_num = part_num * 10000000 + 1
      if part_num == 11:
        end_num = 121225001
      else:
        end_num = (part_num + 1) * 10000000 + 1  

      for i in xrange(begin_num, end_num, 25000):
        start_time = float(time.time())
        # get input compounds
        in_file = "Compound_" + "{:0>9}".format(i) + "_" + "{:0>9}".format(i + 24999) + ".pkl"
        if not os.path.exists(os.path.join(pkl_dir, in_file)):
          logfile.write("%s\t0\tnot exists" % in_file)
          continue
        infile = open(os.path.join(pkl_dir, in_file), "rb")
        compds = cPickle.load(infile).astype(numpy.float32)
        infile.close()
        for target in target_list:
          prediction_dict[target] = sess.run(tf.argmax(softmax_dict[target], 1), feed_dict = {input_placeholder: compds})

        # stack prediction result into a matrix with shape = (num_compds, num_targets)
        prediction = numpy.vstack([prediction_dict[k] for k in target_list]).T
        logfile.write("%s\t%d\n" % (in_file, prediction.sum()))
        # convert into sparse matrix
        if not prediction.sum()==0:
          sparse_prediction = sparse.csr_matrix(prediction)
          # save result into file
          out_file = in_file.replace("pkl", "prediction")
          outfile = open(os.path.join(prediction_dir, out_file), "wb")
          cPickle.dump(sparse_prediction, outfile, protocol=2)
          outfile.close()
          logfile.write(str(sparse_prediction)+"\n")
        print("%s\t%d\t%.3f" % (in_file, prediction.sum(), time.time()-start_time))  
  logfile.write("virtual screen %d ends at: %s\n" % (part_num, datetime.datetime.now()))
  logfile.close()




if __name__ == "__main__":

  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  virtual_screening(target_list, 11)
