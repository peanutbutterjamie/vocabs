#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import os
import re
from tqdm import tqdm 
import pickle
import sys 
import numpy as np
from collections import Counter


# ## Dataframe import

# In[5]:


def create_merged_df():
    with open('./resource_table_new.pickle', 'rb') as f:
        textbook_df = pickle.load(f)
    with open('./story_table_text_new.pickle', 'rb') as f:
        story_df = pickle.load(f)
    with open('./sentence_list_new.pickle', 'rb') as f:
        sentence_df = pickle.load(f)
    with open('./STID_XTID_new.pickle', 'rb') as f :
        STID_XTID=pickle.load(f)
    with open('./STID_ULID_new.pickle', 'rb') as f :
        STID_ULID=pickle.load(f)
    with open('./XTID.pickle', 'rb') as f :
        token_xpos=pickle.load(f)
    with open('./ULID.pickle', 'rb') as f :
        lemma_upos=pickle.load(f)
    st_lemma=pd.merge(STID_ULID,lemma_upos, on='UL ID', how='left')
    st_lemma_txt=pd.merge(st_lemma,sentence_df, on='ST ID', how='left' )
    st_lemma_txt=pd.merge(st_lemma_txt, story_df[['RS ID', 'SR ID', 'SR Code','SR Title']], on='SR ID', how='left')
    st_lemma_txt=pd.merge(st_lemma_txt,textbook_df[['RS ID', 'RS Code','Title']], on='RS ID', how='left' )
    st_lemma_txt.rename(columns={'Title':'RS Title'}, inplace=True)
    st_lemma_txt=st_lemma_txt[['UL ID', 'UPOS:Lemma', 'RS ID', 'RS Code', 'RS Title',
                 'SR ID', 'SR Code', 'SR Title', 'ST ID', 'Sentence' ]]
    return st_lemma_txt




def get_frequency (st_lemma_txt) :  
    word=input("enter the UPOS:Lemma : ").replace(" ", "")
    df_lem= st_lemma_txt[st_lemma_txt['UPOS:Lemma']==word].reset_index(drop=True)
    lemma_freq=len(df_lem)
    st_id_unique=list(df_lem['ST ID'].unique())
    tb_id_unique=list(df_lem['RS ID'].unique())
    sr_id_unique=list(df_lem['SR ID'].unique())
    
    
    tb_title={}
    tt_to_sr={}
    for i in range(len(tb_id_unique)):
        tbid=str(tb_id_unique[i])
        tt=str(df_lem[df_lem['RS ID']==tbid]['RS Code'].unique()[0])
        story_num=list(df_lem[df_lem['RS ID']==tbid]['SR ID'].unique())
        if i == 0 :
            tb_title={tbid:tt}
            tt_to_sr={tbid : story_num}
        else:
            tb_title[tbid]=tt
            tt_to_sr[tbid]= story_num
    
    for tb_num in tb_title:
        tbname=tb_title.get(tb_num)
        srnum=tt_to_sr.get(tb_num)
    
    print(word + " is used " + str(lemma_freq) + " times in " + str(len(st_id_unique)) + " sentences.")
    print(word + " appeared in " + str(len(sr_id_unique)) + " stories : " +', '.join(tb_title.values())) 
    #+ str(len(tb_id_unique)) + " textbooks"
    #return lemma_freq, st_id_unique, tb_id_unique, sr_id_unique



def get_sentence (st_lemma_txt) :
    word=input("enter the UPOS:Lemma : ").replace(" ", "")
    df_lem= st_lemma_txt[st_lemma_txt['UPOS:Lemma']==word].reset_index(drop=True)
    lemma_freq=len(df_lem)
    st_id_unique=list(df_lem['ST ID'].unique())
    #st_txt_unique=list(df_lem['Sentence'].unique())
    tb_id_unique=list(df_lem['RS ID'].unique())
    sr_id_unique=list(df_lem['SR ID'].unique())
    print(" ")
    print(word + " is used in " + str(len(st_id_unique)) + " sentences.")
    print(" ")
    for i in range(len(st_id_unique)) :
        sent=''.join(df_lem[df_lem['ST ID']==st_id_unique[i]]['Sentence'])
        print(str(i+1)+": " + sent)
        print("  ")

    #print(word + " is used " + str(lemma_freq) + " times in " + str(len(st_id_unique)) + " sentences.")
    #print(word + " appeared in " + str(len(sr_id_unique)) + " stories in " +', '.join(tb_title.values())) 
    #+ str(len(tb_id_unique)) + " textbooks"
    #return lemma_freq, st_id_unique, tb_id_unique, sr_id_unique


def get_sentence_story_textbook (st_lemma_txt) :
    with open('./story_table_text_new.pickle', 'rb') as f: story_df = pickle.load(f)
    word=input("enter the UPOS:Lemma : ").replace(" ", "")
    df_lem= st_lemma_txt[st_lemma_txt['UPOS:Lemma']==word].reset_index(drop=True)
    
    lemma_freq=len(df_lem)
    st_id_unique=list(df_lem['ST ID'].unique())
    st_txt_unique=list(df_lem['Sentence'].unique())
    tb_id_unique=list(df_lem['RS ID'].unique())
    sr_id_unique=list(df_lem['SR ID'].unique())
    
    tb_title={}
    tt_to_sr={}
    lem_story={}
    sr_title={}

    
    for i in range(len(tb_id_unique)):
        tbid=tb_id_unique[i]
        tt=str(df_lem[df_lem['RS ID']==tbid]['RS Code'].unique()[0])
        story_num=list(df_lem[df_lem['RS ID']==tbid]['SR ID'].unique())
        if i == 0 :
            tb_title={tbid:tt}
            tt_to_sr={tbid : story_num}
        else:
            tb_title[tbid]=tt
            tt_to_sr[tbid]= story_num
    print(" ")
    print(word + " is used " + str(lemma_freq) + " times in " + str(len(st_id_unique)) + " sentences.")
    print(word + " appeared in " + str(len(sr_id_unique)) + " stories.")
    print(word + " appeared in " + str(len(tb_title.values())) + " resources.")
    print( ', '.join(tb_title.values()))

    print("  ")
    print("========================================================")
    print("  ")
    for i in range(len(sr_id_unique)):
        srid=sr_id_unique[i]
        title=''.join(story_df[story_df['SR ID']==srid]['SR Code'])
        if i ==0 :
            lem_story={sr_id_unique[i]: list(df_lem[df_lem['SR ID']==sr_id_unique[i]]['Sentence'].unique())}
            sr_title={srid:title}
        else:
            lem_story[sr_id_unique[i]] =  list(df_lem[df_lem['SR ID']==sr_id_unique[i]]['Sentence'].unique()) 
            sr_title[srid]=title
    for tb_num in tb_title:
        tb_name=tb_title.get(tb_num)
        srnum=tt_to_sr.get(tb_num)
        for i in srnum:
            story_title=sr_title[i]
            sents=lem_story[i]
            print("Textbook : " + tb_name)
            print("Story : " + story_title)
            print(" ")
            print("Number of sentences with "+ word +" in this story : "  + str(len(sents)))
            print(" ")
    
            if len(sents) > 1 :
                for num in range(len(sents)):
                    sent= "".join(sents[num])
                    print (str(num+1)+ ") " + str(sent))
                    print(" ")
            else: print(str(1)+") "+""+ ''.join(sents))
            print(" ")
            print("========================================================")
            print(" ")


def count_story (st_lemma_txt) :
    
    word=input("enter the UPOS:Lemma : ").replace(" ", "")
    df_lem= st_lemma_txt[st_lemma_txt['UPOS:Lemma']==word].reset_index(drop=True)  
    sr_id_unique=list(df_lem['SR ID'].unique())    
    sr_title={}

    for i in range(len(sr_id_unique)):
        srid=sr_id_unique[i]
        srcode=''.join(story_df[story_df['SR ID']==srid]['SR Code'])
        if i ==0 :
            sr_title={srid:srcode}
        else:
            sr_title[srid]=srcode
    print(" ")
    print("-Story count : " + str(len(sr_id_unique)))
    print("-Story title : \n " + ', \n '.join(sr_title.values()))




def count_textbook (st_lemma_txt) :
    
    word=input("enter the UPOS:Lemma : ").replace(" ", "")
    df_lem= st_lemma_txt[st_lemma_txt['UPOS:Lemma']==word].reset_index(drop=True)  

    tb_id_unique=list(df_lem['RS ID'].unique())
    tb_title={}

    
    for i in range(len(tb_id_unique)):
        tbid=tb_id_unique[i]
        tbcode=''.join(df_lem[df_lem['RS ID']==tbid]['RS Code'].unique())
        if i == 0 :
            tb_title={tbid:tbcode}
        else:
            tb_title[tbid]=tbcode
    print(" ")
    print("RS count : " + str(len(tb_id_unique)))
    print("RS title : \n " + ', \n '.join(tb_title.values()))


    #print(word + " is used " + str(lemma_freq) + " times in " + str(len(st_id_unique)) + " sentences.")
    #print(word + " appeared in " + str(len(sr_id_unique)) + " stories in " +', '.join(tb_title.values())) 
    #+ str(len(tb_id_unique)) + " textbooks"
    #return lemma_freq, st_id_unique, tb_id_unique, sr_id_unique
