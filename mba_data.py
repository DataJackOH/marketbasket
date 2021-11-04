#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# Import dependencies
import streamlit as st
import streamlit.components as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
import pathlib


# An alias for our state
state = st.session_state


# A function to easily go from one step to another
def change_step(next_step):
    state.step = next_step


# Let's initialize our session state
if "data" not in state:
    state.data = []
    state.step = "init"

# Page config
st.set_page_config(page_title="Market Basket Analysis", layout="wide")


# Step 1
if state.step == "init":
    st.button("Load data", on_click=change_step, args=["load"])


uploaded_file = st.sidebar.file_uploader('Upload a file with some order data', type=['csv','xlsx'])

demo = st.sidebar.checkbox('Use demo data')




## add in context


# Info
with st.beta_expander("What is market basket analysis?", expanded=False):
    st.write('Market basket analysis is useful for finding hidden associations between orders.')
    st.write("The key metrics are lift and confidence. Lift is a measure of")


    


if uploaded_file is not None:
#read csv
    try: 
        df =pd.read_csv(uploaded_file)
    except:
        df =pd.read_excel(uploaded_file)
elif demo:
    df = pd.read_csv('https://raw.githubusercontent.com/DataJackOH/marketbasket/main/orderdataset.csv')
else:
    st.title('Market Basket Analysis')
    st.header("""First, upload some data via the sidebar.""" )
    st.header("""This should be a csv or xslx file with one row per order.""")
    st.header("""The only fields you need are 
    1. Item Description
    2. Quantity
    3. A common Order/InvoiceID""")
    st.stop()
    

# In[2]:

st.title('Market Basket Analysis')
st.header("Nice one! Now lets map your data to the columns we're expecting")

st.subheader("""



""")

col1, col2, col3 = st.beta_columns(3)
with st.form(key='columns_in_form'):
        product_description = col1.selectbox('The column which contains your product description', df.columns)
        order_number = col2.selectbox('The column which contains your order/invoice number.', df.columns)
        quantity = col3.selectbox('The column which contains the quantity of the product within that order', df.columns)
        #orderdate = col1.selectbox('The column which contains the date of the order', df.columns)
        submit_button = st.form_submit_button(label='Run the numbers!')


if not submit_button:
    st.stop()
st.success('Boom!')

@st.cache
def clean_up(df, description=product_description, orders=order_number):
    df[description] = df[description].str.strip()
    df.dropna(axis=0, subset=[orders], inplace=True)
    df[orders] = df[orders].astype('str')
    return df



# In[5]:


df = clean_up(df=df)

 

# In[6]:


@st.cache
def rearrange_df(df=df, description=product_description, orders=order_number, quantity=quantity):
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
basket= None


# In[10]:


frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
basket_sets = None


# In[11]:
@st.cache
def associationcleanup(frequent_itemsets=frequent_itemsets):
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules["product_a"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules["product_b"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules = rules[['product_a','product_b','support','confidence','lift','antecedent_len']]
    frequent_itemsets = None
    return rules 
    
rules = associationcleanup(frequent_itemsets)


# In[12]:
st.dataframe(rules)


@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csvconvert = convert_df(rules)

st.download_button(
    label="Download data as CSV",
    data=csvconvert,
    file_name='marketbasketanalysis.csv',
    mime='text/csv')
