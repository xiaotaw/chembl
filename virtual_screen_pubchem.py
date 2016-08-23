# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: apply pk model to pubchem dataset, to screen potential active substrate(drugs)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy
import cPickle
import datetime
import tensorflow as tf
from scipy import sparse

import pk_model


def virtual_screening(target_list):

  # virtual screen log file
  log_dir = "log_files"
  logpath = os.path.join(log_dir, "virtual_screen_pubchem.log")
  logfile = open(logpath, "w")
  logfile.write("virtual screen starts at: %s\n" % datetime.datetime.now())

  # input and output dir
  pkl_dir = "/raid/xiaotaw/pubchem/pkl_files"
  prediction_dir = "/raid/xiaotaw/pubchem/prediction_files"
  if not os.path.exists(prediction_dir):
    os.mkdir(prediction_dir)

  # get input compds
  filename_list = glob.glob(pkl_dir + "/*.pkl")

  # screening
  with tf.Graph().as_default(), tf.device("/gpu:1"):
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
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      # Restores variables from checkpoint
      saver.restore(sess, "ckpt_files/model.ckpt-40000")

      for in_file in filename_list:
        infile = open(in_file, "rb")
        compds = cPickle.load(infile).astype(numpy.float32)
        infile.close()

        for target in target_list:
          prediction_dict[target] = sess.run(
            tf.argmax(softmax_dict[target], 1), 
            feed_dict = {input_placeholder: compds})

        # stack prediction result into a matrix with shape = (num_compds, num_targets)
        prediction = numpy.vstack([prediction_dict[k] for k in target_list]).T
        logfile.write("%s: %d\n" % (in_file, prediction.sum())
        # convert into sparse matrix
        if not prediction.sum() == 0:
          sparse_prediction = sparse.csr_matrix(prediction)
          # save result into file
          outfile = open(out_file, "wb")
          cPickle.dump(sparse_prediction, outfile, protocol=2)
          outfile.close()
          logfile.write(str(sparse_prediction)+"\n")

  logfile.close()




if __name__ == "__main__":

  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr",
                 "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]

  virtual_screening(target_list)
