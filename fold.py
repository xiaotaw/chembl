# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: 

import os

import pk_input
import pk_train
import pk_eval


batch_size = 128
log_dir = "log_files"
ckpt_dir = "cpkt_test"
target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]




def test_model():
  if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir) 
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  train_log_filename = os.path.join(log_dir, "train.log")
  eval_log_filename = os.path.join(log_dir, "eval.log")

  dataset_dict = {}
  for target in target_list:
    dataset_dict[target] = pk_input.get_inputs_pseudo(target, True)

  neg_dataset = pk_input.get_inputs_pseudo("pubchem_neg_sample", False)

  # create tmp checkpoint dir
  if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
  
  
  # train fcnn model
  pk_train.train(dataset_dict, neg_dataset, target_list, train_log_filename, batch_size=batch_size,
    ckpt_dir = ckpt_dir, train_from = 0, max_step = 2000, 
    pretrained_variables = False,
    start_learning_rate = 0.0, decay_step = 8000, decay_rate = 0.7)

  """
  # eval model
  pk_eval.evaluate(dataset_dict, neg_dataset, target_list, eval_log_filename,
    ckpt_dir = ckpt_dir)
  """
 

def fold(target_list, i):

  ckpt_dir = "tmp"+str(i)
  train_log_filename = "train_"+str(i)+".log"
  eval_log_filename = "eval_"+str(i)+".log"

  train_compd_dict, train_label_dict = get_input(target_list, i)

  # create tmp checkpoint dir
  if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
  
  # train fcnn model
  train(train_compd_dict, train_label_dict, target_list, train_log_filename, 
        ckpt_dir = ckpt_dir, train_from = 0, max_step = 700, 
        pretrained_variables = "protein_kinase_model.ckpt-3601",
        start_learning_rate = 0.1, decay_step = 16000, decay_rate = 0.7)
  
def fold_cv(target_list, i):

  ckpt_dir = "tmp"+str(i)
  eval_log_filename = "eval_"+str(i)+".log"

  cv_compds_dict, cv_labels_dict = get_cv_input(target_list, i)

  # create tmp checkpoint dir
  if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

  
  # wval fcnn model
  evaluate(cv_compds_dict, cv_labels_dict, target_list, eval_log_filename, ckpt_dir = ckpt_dir)


if __name__ == "__main__":



  #fold(target_list, 0)
  test_model()
  
 
  
  
  


