#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
# import pyterrier as pt
# if not pt.started():
#     pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"], mem=8000)
import pyterrier as pt
if not pt.started():
    pt.init(mem=32000)
#set pyterrier's run cache directory to be in your /users/tr.$USER space. Adjust if necessary
pt.cache.CACHE_DIR = "/nfs/pyterrier_transformer_cache" 


# In[2]:


import pandas as pd


# In[4]:


import json


# In[3]:


get_ipython().system('ls data/all_*')


# In[6]:


all_data = []
with open('data/all_blocks.txt', 'r') as f:
    for l in f:
        json_data = json.loads(l)
        all_data.append(json_data)


# In[7]:


len(all_data)


# In[8]:


all_data[0]


# In[9]:


orquac = pd.DataFrame(all_data)


# In[10]:


get_ipython().system('rm -rf ./index_orquac')
pd_indexer = pt.DFIndexer("./index_orquac")
indexfull = pd_indexer.index(text=orquac["text"],  docno=orquac["id"], title=orquac['title'])


# In[ ]:




