#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import clear_output 
import stanza
import pandas as pd
import os
import re
from tqdm import tqdm 
import pickle
import sys 
import numpy as np
from collections import Counter



def create_merged_df():
    with open('./data/resource_table_new.pickle', 'rb') as f:
        textbook_df = pickle.load(f)
    with open('./data/story_table_text_new.pickle', 'rb') as f:
        story_df = pickle.load(f)
    with open('./data/sentence_list_new.pickle', 'rb') as f:
        sentence_df = pickle.load(f)
    with open('./data/STID_XTID_new.pickle', 'rb') as f :
        STID_XTID=pickle.load(f)
    with open('./data/STID_ULID_new.pickle', 'rb') as f :
        STID_ULID=pickle.load(f)
    with open('./data/XTID.pickle', 'rb') as f :
        token_xpos=pickle.load(f)
    with open('./data/ULID.pickle', 'rb') as f :
        lemma_upos=pickle.load(f)
    st_lemma=pd.merge(STID_ULID,lemma_upos, on='UL ID', how='left')
    st_lemma_txt=pd.merge(st_lemma,sentence_df, on='ST ID', how='left' )
    st_lemma_txt=pd.merge(st_lemma_txt, story_df[['RS ID', 'SR ID', 'SR Code','SR Title']], on='SR ID', how='left')
    st_lemma_txt=pd.merge(st_lemma_txt,textbook_df[['RS ID', 'RS Code', 'Title']], on='RS ID', how='left' )
    st_lemma_txt.rename(columns={'Title':'RS Title'}, inplace=True)
    st_lemma_txt=st_lemma_txt[['UL ID', 'UPOS:Lemma', 'RS ID', 'RS Code', 'RS Title',
                 'SR ID', 'SR Code', 'SR Title', 'ST ID', 'Sentence' ]]
    return st_lemma_txt




def get_sr_st_count(ul_df):
    st_lemma_txt=create_merged_df()

    sr_count=[]
    st_count=[]
    st_div_sr=[]
    
    for word in tqdm(list(ul_df['UL ID'])):
        df_lem= st_lemma_txt[st_lemma_txt['UL ID']==word].reset_index(drop=True)
        sr=df_lem['SR ID'].nunique()
        st=df_lem['ST ID'].nunique()
        sr_count.append(sr)
        st_count.append(st)
        if sr!=0:
            st_div_sr.append(round(st/sr, 2))
        else : st_div_sr.append(0)
        
    ul_df['SR Count']=sr_count
    ul_df['ST Count']=st_count
    ul_df['ST/SR']=st_div_sr
    return ul_df





def reference_words_exclusion (ref_df, lemma_df, condition) :
    if condition == 'LEMMA' : 
        lemma_df['entry'], lemma_df['POS'] = lemma_df['UPOS:Lemma'].apply(lambda x: re.split(":", x)[1]), lemma_df['UPOS:Lemma'].apply(lambda x: re.split(":", x)[0])
        lemma_df=lemma_df[['entry', 'POS', 'UPOS:Lemma', 'UL ID' ]]
        idx_drop=[]
        for entry in list(ref_df['Lemma']):
            if entry in list(lemma_df['entry']):
                for e in list(lemma_df[lemma_df['entry']==entry].index): idx_drop.append(e)
            else: continue

        entry_df2=lemma_df.drop(lemma_df.index[idx_drop]).reset_index(drop=True)

        return entry_df2[['UL ID','UPOS:Lemma']]
    
    elif condition =='POS:LEMMA' : 
        idx_drop=[]
        for lemma in list(ref_df['POS:LEMMA']) :
            if lemma in list(lemma_df['UPOS:Lemma']) : 
                e=lemma_df[lemma_df['UPOS:Lemma']==lemma].index[0]
                idx_drop.append(e)
            else: continue
        entry_df2 = lemma_df.drop(index=idx_drop, axis=0).reset_index(drop=True)[['UPOS:Lemma', 'UL ID']]
        return entry_df2
    else : print("input error. enter either \"LEMMA\" or \"POS:LEMMA\ " )


