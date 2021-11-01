#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# Import dependencies
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
import pathlib

uploaded_file = st.file_uploader('Choose a file', type=['csv','xlsx'])

if uploaded_file is not None:
#read csv
    try: 
        df =pd.read_csv(uploaded_file)
    except:
        df =pd.read_excel(uploaded_file)
else:
    st.warning('you need to upload a csv or excel file.')

# In[2]:

st.title('Market Basket Analysis')



def clean_up(df, description='Description', orders='InvoiceNo'):
    df[description] = df[description].str.strip()
    df.dropna(axis=0, subset=[orders], inplace=True)
    df[orders] = df[orders].astype('str')
    return df


# In[5]:


clean_up(df=df)


# In[6]:


def rearrange_df(df=df, description='Description', orders='InvoiceNo', quantity='Quantity'):
    basket = (df
          .groupby([orders, description])[quantity]
          .sum().unstack().reset_index().fillna(0)
          .set_index(orders))
    return basket


# In[7]:


basket = rearrange_df(df=df)


# In[8]:



# In[9]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
basket= None


# In[10]:


frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
basket_sets = None


# In[11]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules = rules[['antecedents','consequents','support','lift']]
frequent_itemsets = None


# In[12]:

st.write(rules)

# In[ ]:




