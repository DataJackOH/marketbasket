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

# Page config
st.set_page_config(page_title="Market Basket Analysis", layout="wide")



uploaded_file = st.sidebar.file_uploader('Upload a file with some order data', type=['csv','xlsx'])

demo = st.sidebar.checkbox('Use demo data')



st.title('Market Basket Analysis 🛒 ')

## add in context


# Info
with st.expander("What is market basket analysis?", expanded=False):
    st.write("""Market basket analysis is an algorithim which finds items that are commonly purchased together.
                
This information can then be used to cross/upsell.
                
The typical story is Walmart ran this and found a hidden, strong association between Beer and Nappies. 

They put Beer and Nappies on a cross-promotion and made a killing! 🍺 🍼""")


with st.expander("How do I use this tool?", expanded=False): 
    st.write("""You have 2 options to upload data:""")
    st.write(""" 
    -   Upload a CSV/xslx of your own **Customer order data** via the sidebar.
        - Your file must contain at least three columns:
            -   A 'Product Description' column
            -   An 'Order Number' or common ID column that's shared across different products within the same order.
            -   A 'Quantity' column, showing how many of that product was purchased in the order.""")

    st.write("""


    - Use the 'Demo' order data.
        - Open the sidebar and tick 'Use demo data'
    """)
    st.write("""
    - You can **download** the analysis once it's finished, with the "Download data as csv" button
    """)
               


with st.expander("What does the table mean?", expanded=False): 
  st.write("""
    -   The key metrics are **Lift** and **Confidence**:
        -   **Lift** is a measure of how frequently your items occur together in baskets. 
            -   Where 1 is completely random, <1 is negative (or less likely than random) and >1 is a positive association (more likely than random). 
        -   **Confidence** is directional, and is the percent of transactions of product_A which also include product_B.""")

if uploaded_file is not None:
#read csv
    try: 
        df =pd.read_csv(uploaded_file)
    except:
        df =pd.read_excel(uploaded_file)
elif demo:
    df = pd.read_csv('https://raw.githubusercontent.com/DataJackOH/marketbasket/main/orderdataset.csv')
else:
    st.stop()
    

# In[2]:

st.subheader("Nice one! Now lets map your data to the columns we're expecting")

st.subheader("""



""")

col1, col2, col3 = st.columns(3)
with st.form(key='columns_in_form'):
        product_description = col1.selectbox('The column which contains your product description', df.columns)
        order_number = col2.selectbox('The column which contains your order/invoice number.', df.columns)
        quantity = col3.selectbox('The column which contains the quantity of the product within that order', df.columns)
        #orderdate = col1.selectbox('The column which contains the date of the order', df.columns)
        submit_button = st.form_submit_button(label='Run the numbers!')



if not submit_button:
    st.stop()


@st.cache
def clean_up(df, description=product_description, orders=order_number):
    df[description] = df[description].str.strip()
    df.dropna(axis=0, subset=[orders], inplace=True)
    df[orders] = df[orders].astype('str')
    return df

df = clean_up(df=df)

# In[5]:



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
    rules["consequent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules = rules[rules['antecedent_len'] == 1]
    rules = rules[rules['consequent_len'] == 1]
    rules = rules[['product_a','product_b','support','confidence','lift']]
    rules = rules.sort_values(by='lift', ascending=False)
    frequent_itemsets = None
    return rules 



try: 
    rules = associationcleanup(frequent_itemsets)
    st.dataframe(rules)
except:
        st.error('Please enter a valid input')
        st.stop()
    
st.success("""Boom! - your data has been sorted so the top associations (**greatest lift**) appear first.""")

# In[12]:

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
