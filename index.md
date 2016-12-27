# Abstract
Author: xiaotaw@qq.com (Any bug report is welcome)
Time: Aug 2016
Addr: Shenzhen
Description: This is a deep neural network model, which predicts potential drugs for protein kinase targets. 
Website: https://xiaotaw.github.io/chembl/


# Background
  (add background for using DNN to build this qsar model)

# Problem
  (add one sentence abstract for current challenge)

# Solution
  (how we solve the problem)

# Method
## 1 get data
     1.1 positive dataset was downloaded from chembl database
     1.2 negtive dataset was selected from pubchem database(based on a reasonable assumption that almost the compound in pubchem was NOT the substrate of a protein kinase)

## 2 build the model
     2.1 deep neural network was used(based on tensorflow)
     2.2 a 'Tree' comprises one 'Term' and several 'Branches', where the 'Term' extracts the mutual figures of all the protein kinase.

## 3 train and evaluation
     3.1 we train the model jointly and apply the model on pubchem dataset for virtual screening.
 
