# Abstract
Author: xiaotaw@qq.com (Any bug report is welcome)

Time Created: Aug 2016

Time Updated: Dec 2016

Addr: Shenzhen, China

Description: We attempt to explore ChEMBL's Inhibitors by deep neural network

Website: https://xiaotaw.github.io/chembl/


# Background
  (add background for using DNN and RF to build this qsar model)

# Problem
  (add one sentence abstract for current challenge)

# Solution
  (how we solve the problem)

# Method

## 1 get data
     1.1 positive dataset was downloaded from chembl database
     1.2 negtive dataset was selected from pubchem and chembl database(based on a reasonable assumption that almost the compound in pubchem was NOT the substrate of a protein kinase)

## 2 build the model
     2.1 deep neural network(based on tensorflow)
     2.2 random forest(based on scikit-learn)
     2.3 a 'Tree' comprises one 'Term' and several 'Branches', where the 'Term' extracts the mutual figures of all the protein kinase.

## 3 train and evaluation
     3.1 we train the model seperately and jointly, and then apply the model on pubchem dataset for virtual screening.
 

