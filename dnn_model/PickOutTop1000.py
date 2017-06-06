
# coding: utf-8

# In[16]:

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


# In[17]:

target_list = ["CHEMBL203", "CHEMBL204", "CHEMBL235", "CHEMBL236", 
               "CHEMBL244", "CHEMBL260", "CHEMBL4805", "CHEMBL4822"]

g_step_list = [2161371, 2236500, 2086841, 2236500, 
               2161951, 2252100, 2168041, 1936221]

# In[18]:

ChemDiv_dir = "/raid/xiaotaw/ChemDiv/"
fn_list = ["DC01_350000.sdf", "DC02_350000.sdf", 
           "DC03_222773.sdf", "DC_saltdata_not-available_124145.sdf", 
           "IC_non-excl_82693.sdf", "NC_340320.sdf"]
#sup0 = Chem.SDMolSupplier(ChemDiv_dir + fn_list[0])
#ms0 = [x for x in sup0 if x is not None]


# In[19]:

#sup1 = Chem.SDMolSupplier(ChemDiv_dir + fn_list[1])
#ms1 = [x for x in sup1 if x is not None]
#sup2 = Chem.SDMolSupplier(ChemDiv_dir + fn_list[2])
#ms2 = [x for x in sup2 if x is not None]


# In[20]:

i = 7
target = target_list[i]
g_step = g_step_list[i]
pred_dir = "/home/scw4750/Documents/chembl/dnn_model/pred_files/%s/" % target
pred_fn = pred_dir + "vs_chemdiv_%s_128_0.800_4.000e-03_%d.pred1000" % (target, g_step)
chemdiv_pred = pd.read_csv(pred_fn, sep="\t", index_col=0, names=["id", "pred"])
#chemdiv_pred
id_list = chemdiv_pred["id"].values
#id_list


# In[23]:

m1000 = []
for fn in fn_list:
    print("start %s" % fn)
    sup = Chem.SDMolSupplier(ChemDiv_dir + fn)
    for m in sup:
        if (m is not None) and (m.GetProp("IDNUMBER") in id_list):
            m1000.append(m)
            #print(m.GetProp("IDNUMBER"))        
    print("finished %s" % fn)


# In[38]:

def get_pred_value(id_):
  return chemdiv_pred["pred"][chemdiv_pred["id"] == id_].values[0]

m1000.sort(key=lambda x: get_pred_value(x.GetProp("IDNUMBER")), reverse=True)


# In[40]:

writer = Chem.SDWriter(pred_fn.replace(".pred1000", "_top1000.sdf"))
for m in m1000:
    writer.write(m)
    


# In[ ]:



