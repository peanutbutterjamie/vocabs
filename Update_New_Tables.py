#!/usr/bin/env python
# coding: utf-8



from IPython.display import clear_output 
import stanza
import pandas as pd
import os
import re
from tqdm import tqdm 
import pickle
import sys 
import numpy as np
get_ipython().system('pip install nltk')
from nltk import sent_tokenize
get_ipython().system('pip install spacy')
import spacy 
get_ipython().system('pip install stanza')
nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma')
clear_output()



def update_resource_num (srpath, class_name):
    with open('./data/resource_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)
    resource_title=re.split("/", str(srpath))[-1] 
    # resource_code (ENG), series(ESC), label(4), gr(4), title(Esacalte 4), rscode (ESC-4)
    test_info=re.split('-', resource_title)
    resource_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+label
    
    # 새로운 레퍼런스 정보 업데이트용 리스트
    new_resource_info=[]

    # 새로운 Rf id 부여 
    lastid=textbook_df['RS ID'].nunique()
    resourceidnumber="RS" + str(lastid+1).zfill(3)

    
    # 리스트 업데이트 
    new_resource_info.append([resourceidnumber, class_name, resource_code, series, gr, label, rscode, title])
    
    # Df 로 변환 후 기존 테이블과 합친 후 저장 
    new_rf=pd.DataFrame(new_resource_info, columns=textbook_df.columns)
    resource_df_new=pd.concat([textbook_df, new_rf]).reset_index(drop=True)
    with open('./data/resource_table_new.pickle', 'wb') as f : 
        pickle.dump(resource_df_new, f)
    print("------------"+resource_title+"------------")
    print("resource table updated :"+ resourceidnumber)

    
def get_resource_table(class_name, resource_information):
    path="/Users/jamielee/Library/CloudStorage/Dropbox/AIEI - 코퍼스/코퍼스 리소스/리소스 TXT/TXT_"
    srpath=path+class_name+"/"+resource_information
    with open('./data/resource_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)
    # 레퍼런스 제목
    resource_title=re.split("/", srpath)[-1] 
    # resource_code (ENG), series(ESC), label(4), gr(4), title(Esacalte 4), rscode (ESC-4)
    test_info=re.split('-', resource_title)
    resource_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+label
    # <resource table update>
    # 레퍼런스 테이블에 해당 레퍼런스가 존재하는지 확인
    if rscode in list(textbook_df['RS Code']) : 
        print(rscode + " is already in the resource table: "+ ''.join(textbook_df[textbook_df['RS Code']==rscode]['RS ID']))
    # 없는 경우 새로운 RS ID 를 부여하여 테이블 업데이트 
    else:
        update_resource_num(srpath, class_name)




def get_story_ids(srpath, srlist, rsid, rscode, last_srid_num):
    with open('./data/story_table_text_new.pickle', 'rb') as f: story_df = pickle.load(f)
    new_story_info=[]
    for i in range(len(srlist)):
        try : 
            f = open(srpath+"/"+srlist[i], "r", encoding = 'utf-8')
            sr_text = ''.join(f.readlines())
            f.close()
        except:
            f = open(srpath+"/"+srlist[i], "r", encoding = 'cp949')
            sr_text = ''.join(f.readlines())
            f.close()
            
        new_sr_id="SR" + str(last_srid_num+1+i).zfill(4)

        sr_info=re.split('-', srlist[i])
        unit=sr_info[0]
        group=sr_info[1]
        sr_order=sr_info[2]
        if '.txt' in sr_info[3]:
            sr_title=re.split(".txt", sr_info[3])[0]
        else:
            sr_title=sr_info[3]

        sr_code=rscode+"-"+unit+"-"+sr_order

        new_story_info.append([rsid, new_sr_id, sr_code, group, sr_order, unit, sr_title, sr_text])

    story_df2=pd.DataFrame(new_story_info, columns=story_df.columns)
    return story_df2





def check_stories(path, srlist, srdf2, rsid, rscode, sr_idx):
    for i in range(len(srlist)): 
        try : 
            f = open(path+"/"+srlist[i], "r", encoding = 'utf-8')
            sr_text = ''.join(f.readlines())
            f.close()
        except:
            f = open(path+"/"+srlist[i], "r", encoding = 'cp949')
            sr_text = ''.join(f.readlines())
            f.close()
            
        sr_info=re.split('-', srlist[i])

        unit, group, sr_order=sr_info[0],sr_info[1], sr_info[2]
        sr_title=re.split(".txt", sr_info[3])[0]
        sr_code=rscode+"-"+unit+"-"+sr_order
        
        indx=list(srdf2[srdf2['SR Code']==sr_code].index)
        if len(indx) >1 : 
            print("duplicated stories")
            print(srlist)
            break
        elif len(indx) == 1:
            srdf2.iloc[indx[0]]['Text']=sr_text
    with open('./data/story_table_text_new.pickle', 'wb') as f : pickle.dump(srdf2, f)
    print("story table text updated to : ./data/story_table_text_new.pickle")
    story_df2=srdf2[sr_idx[0]:sr_idx[-1]]
    return story_df2



def udpate_story_table(class_name, resource_information):
    
    path="/Users/jamielee/Library/CloudStorage/Dropbox/AIEI - 코퍼스/코퍼스 리소스/리소스 TXT/TXT_"
    srpath=path+class_name+"/"+resource_information
    
    resource_title=re.split("/", srpath)[-1] 
    test_info=re.split('-', resource_title)
    resource_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+label
    
    
    story_list = sorted(os.listdir(srpath))
    if '.DS_Store' in story_list : story_list.remove('.DS_Store')

    with open('./data/story_table_text_new.pickle', 'rb') as f: story_df = pickle.load(f)
    with open('./data/resource_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)
    rsid=''.join(textbook_df[textbook_df['RS Code']==rscode]['RS ID'])
    if rsid not in set(story_df['RS ID']) : 
        #story_df= pd.read_excel('./data/story_table_text.xlsx', engine='openpyxl', index_col=0)
        last_srid_num=story_df['SR ID'].nunique()
        story_df2=get_story_ids(srpath, story_list, rsid, rscode, last_srid_num)
        
        story_df_new=pd.concat([story_df, story_df2]).reset_index(drop=True)
        with open('./data/story_table_text_new.pickle', 'wb') as f : 
            pickle.dump(story_df_new, f)
        
        print(resource_title +" : story table updated  - "+ rscode)
        print("updated stories :"+ str(len(story_df2)))
        return story_df_new
       
    elif rsid in set(story_df['RS ID']) :     
        #story_df= pd.read_excel('./data/story_table_text.xlsx', engine='openpyxl', index_col=0)
        sr_idx=list(story_df[story_df['RS ID']==rsid].index)
        
        print("story list already in the table : " + resource_title)
        print("story index : " + str(sr_idx))
        srdf2=story_df.copy()
        story_df2=check_stories(srpath, story_list, srdf2, rsid, rscode, sr_idx)
        return story_df2




def get_st_id(sentencedf) :
    sent_num=[]
    
    with open('./data/sentence_list_new.pickle', 'rb') as f: 
        orig_sent_df= pickle.load(f)
    len_orig_df= len(orig_sent_df)
    for i in range(1, len(sentencedf)+ 1) :
        sent_num.append("ST" + str(len_orig_df+i).zfill(5))
    sentencedf['ST ID'] = sent_num
    
    sentencedfnew=pd.concat([orig_sent_df, sentencedf]).reset_index(drop=True)
    
    
    with open('./data/sentence_list_new.pickle', 'wb') as f:
        pickle.dump(sentencedfnew, f)
        
    return sentencedf.reset_index(drop=True)

def get_sent_df(storydf):
    df1=pd.DataFrame()
    for i in range(len(storydf)):
        if i ==0 :
            sents=storydf['Text'].iloc[i].split("\n")
            df1=pd.Series(sents).to_frame()
            df1['SR ID']=storydf['SR ID'].iloc[i]
        else:
            sents=storydf['Text'].iloc[i].split("\n")
            df2=pd.Series(sents).to_frame()
            df2['SR ID']=storydf['SR ID'].iloc[i]
            df1=pd.concat([df1, df2])
    result=df1.reset_index(drop=True)
    result.rename(columns = {0:'Sentence'}, inplace=True)
    result=result.replace('', np.nan).dropna(subset=['Sentence'], axis=0).reset_index(drop=True)
    result=result[['SR ID', 'Sentence']].reset_index(drop=True)
    result['Sentence'] = result['Sentence'].map(clean_data)
    result_id_df=get_st_id(result)
    return result_id_df



def clean_data(text):
    res = re.sub("\n", " ", text)     # 줄바꿈 제거
    res = re.sub("\r", " ", res)  
    res = re.sub("	", "", res)      # 탭 제거
    res = re.sub("  ", " ", res)     # 여백 2개를 1개로 제거
    res = re.sub("   ", " ", res)    # 여백 3개를 1개로 제거
    return res


def get_pos (txt, SentenceID,StoryID) :
    list_of_tokens=[]
    txt=nlp(txt)
    for sent in txt.sentences:
        for word in sent.words:
            if word.upos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                list_of_tokens.append([StoryID, SentenceID, word.xpos, word.text, word.xpos + ":" + word.text.lower(),
                                       word.upos, word.lemma, word.upos + ":" + word.lemma.lower() ])
            else: list_of_tokens.append([StoryID, SentenceID, word.xpos, word.text,  word.xpos + ":" + word.text.lower(),
                                         'n/a' , 'n/a','n/a'])
    return list_of_tokens

def tokenize_sentence (data):  
    column_name=['SR ID','ST ID','XPOS','Token','XPOS:Token',
                 'UPOS', 'Lemma', "UPOS:Lemma"]
    df_1=pd.DataFrame()
    for i in range(len(data)):
        if i ==0 : 
            token_list = get_pos(data['Sentence'].iloc[i], 
                                 data['ST ID'].iloc[i],
                                 data['SR ID'].iloc[i], 
                                 )
            df_1=pd.DataFrame(token_list, columns=column_name)
        else: 
            token_list = get_pos(data['Sentence'].iloc[i], 
                                 data['ST ID'].iloc[i],
                                 data['SR ID'].iloc[i], 
                                 )
            df_2=pd.DataFrame(token_list, columns=column_name)
            df_1=pd.concat([df_1, df_2],axis=0).reset_index(drop=True)
    
    return df_1



def get_initial_ids(df):
    tokenized_df=tokenize_sentence(df)
    get_initial_token_xpos_ids(tokenized_df)
    get_initial_lemma_upos_ids(tokenized_df)
    return tokenized_df


def get_intial_xpos_lemma (tokenized_df):
    df_whole, xpos_df=get_token_xpos_ids(tokenized_df)
    df_whole2, lemma_df=get_lemma_upos_ids(df_whole)
    return df_whole2, xpos_df, lemma_df


def get_lemma_upos_ids(df_tokenized) :
    with open('./data/ULID.pickle', 'rb') as f :
        df_lemma_upos=pickle.load(f)
    list_lemma_upos=list(df_lemma_upos['UPOS:Lemma'])
    
    a= len(df_lemma_upos)
    ul_ids=[]

    last_ul_id=int(df_lemma_upos.iloc[-1]['UL ID'][2:]) 

    for i in range(len(df_tokenized)) :
        
        pos_lem = df_tokenized["UPOS:Lemma"].iloc[i]
        
        if pos_lem== 'n/a' : ul_ids.append('n/a')
        
        elif pos_lem in list_lemma_upos: #list 혹은 set -> df에서 변환
            ul_ids.append(''.join(df_lemma_upos.loc[(df_lemma_upos["UPOS:Lemma"] == pos_lem)]['UL ID'].tolist()))
       
        elif pos_lem not in list_lemma_upos:
	    
            new_id= "UL"+str(last_ul_id+1).zfill(5)
            list_lemma_upos.append(pos_lem) 
            df_lemma_upos.loc[len(df_lemma_upos)] = [new_id, pos_lem]
            ul_ids.append(new_id)
            last_ul_id=last_ul_id+1
        else: print("exception : " + \
                    str(pos_lem))
    df_tokenized['UL ID']=ul_ids 
    with open('./data/ULID.pickle', 'wb') as f:
        pickle.dump(df_lemma_upos, f, pickle.HIGHEST_PROTOCOL)   
    print("-newly added upos:lemma IDs -> "+ str(len(df_lemma_upos)-a))
    result_lemma= df_tokenized[['ST ID',"UL ID"]].replace('n/a', np.NaN).dropna(subset=['UL ID'])
    return df_tokenized, result_lemma




def get_token_xpos_ids(df_tokenized):
    with open('./data/XTID.pickle', 'rb') as f :
        df_token_xpos=pickle.load(f)
    list_token_xpos=list(df_token_xpos['XPOS:Token'])
    a=len(df_token_xpos)
    
    xtids=[]
    
    for i in range(len(df_tokenized)):
        
        word=df_tokenized['XPOS:Token'].iloc[i]
        
        if word in list_token_xpos: #list 혹은 set -> df에서 변환
            xtids.append(''.join(df_token_xpos.loc[(df_token_xpos['XPOS:Token'] == word)]['XT ID'].tolist()))
        elif word not in list_token_xpos:
            newxtid="XT"+str(len(df_token_xpos)+1).zfill(5)
            df_token_xpos.loc[len(df_token_xpos)+1] = [newxtid, word]
            list_token_xpos.append(word)
            
            xtids.append(newxtid)
            
    df_tokenized['XT ID']=xtids
    
    with open('./data/XTID.pickle', 'wb') as f:
        pickle.dump(df_token_xpos, f, pickle.HIGHEST_PROTOCOL)
        
    print("-newly added token:xpos IDs -> "+ str(len(df_token_xpos)-a))
    return df_tokenized, df_tokenized[['ST ID','XT ID']]



def get_xpos_upos (df):
    tokenized_df=tokenize_sentence(df)
    df_whole, xpos_df=get_token_xpos_ids(tokenized_df)
    df_whole2, lemma_df=get_lemma_upos_ids(df_whole)
    return df_whole2, xpos_df, lemma_df





def check_duplicate(resource_information):          

    with open('./data/story_table_text_new.pickle', 'rb') as f: story_df = pickle.load(f)
    with open('./data/resource_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)
    with open('./data/sentence_list_new.pickle', 'rb') as f: sentence_df = pickle.load(f)

    #resource_title=re.split("/", srpath)[-1] 
    test_info=re.split('-', resource_information)
    resource_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+label

    rsid=''.join(textbook_df[textbook_df['RS Code']==rscode]['RS ID'])
    
    srids=list(story_df[story_df['RS ID']==rsid]['SR ID'])
    
    if srids[0] in list(sentence_df['SR ID'].unique()):
        print("Stories in the Resource " + rscode + " : "+title+" is already in the Sentence Table")
        return False
    else: return True
    

def get_sentence_table_updated (class_name, resource_information):
    
    path="/Users/jamielee/Library/CloudStorage/Dropbox/AIEI - 코퍼스/코퍼스 리소스/리소스 TXT/TXT_"
    srpath=path+class_name+"/"+resource_information

    with open('./data/story_table_text_new.pickle', 'rb') as f: story_df = pickle.load(f)
    with open('./data/resource_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)
    with open('./data/sentence_list_new.pickle', 'rb') as f: sentence_df = pickle.load(f)
    #resource_title=re.split("/", srpath)[-1] 
    test_info=re.split('-', resource_information)
    resource_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+label
    rsid=''.join(textbook_df[textbook_df['RS Code']==rscode]['RS ID'])
    
    srids=list(story_df[story_df['RS ID']==rsid]['SR ID'])
    if srids[0] in list(sentence_df['SR ID'].unique()):
        print("Resource Story" + rscode + "-"+title+" is already in the Sentence Table")
        exit()
    else: pass
    
    storyindex=list(story_df[story_df['RS ID']==rsid].index)
    new_story_df=story_df[storyindex[0]:storyindex[-1]+1].reset_index(drop=True)

    #sentence 
    sentence_id_df=get_sent_df(new_story_df)

    print("------------"+title+"------------")
    print("sentence table updated :" +str(len(sentence_id_df))+" sentences")

    token_lemma, xpos_df, lemma_df=get_xpos_upos(sentence_id_df)

    st_xt=token_lemma.replace('n/a', np.NaN).dropna(subset=['XT ID'])[['ST ID','XT ID']].reset_index(drop=True)
    with open('./data/STID_XTID_new.pickle', 'rb') as f :
        STID_XTID=pickle.load(f)

    new_xt_st=pd.concat([STID_XTID, st_xt]).reset_index(drop=True)

    with open('./data/STID_XTID_new.pickle', 'wb') as f:
        pickle.dump(new_xt_st, f)

    with open('./data/STID_ULID_new.pickle', 'rb') as f :
        STID_ULID=pickle.load(f)

    st_ul=lemma_df.reset_index(drop=True)

    new_ul_st=pd.concat([STID_ULID, st_ul]).reset_index(drop=True)

    with open('./data/STID_ULID_new.pickle', 'wb') as f:
        pickle.dump(new_ul_st, f)
    print("--XT ID UL ID Updated")



def update_tables(*args):    
    # 스토리 경로 하나씩 처리 
    for class_name, resource_information in tqdm(args):
        get_resource_table(class_name, resource_information)

        udpate_story_table(class_name, resource_information)

        #sentence 
        get_sentence_table_updated(class_name, resource_information)



def update_resource_stories(*args):    
    # 스토리 경로 하나씩 처리 
    for class_name, resource_information in tqdm(args):
        
        # resource table 업데이트
        get_resource_table(class_name, resource_information)
        

        # story table 업데이트
        udpate_story_table(class_name, resource_information)
        
        
        # resource table 중복 및 Null 값 확인 
        with open('./data/resource_table_new.pickle', 'rb') as f:textbook_check= pickle.load(f)
        
        len_dup_rsid=len(textbook_check[textbook_check.duplicated(['RS ID'])])
        len_dup_title=len(textbook_check[textbook_check.duplicated(['Title'])])
        
        # 중복 있을 경우 Loop Break
        if len_dup_rsid + len_dup_title !=0 :
            print("*********RS ID Duplicated : " + str(len_dup_rsid))
            print("*********Title Duplicated : " + str(len_dup_title))
            break
        else : pass   
            
        # Null 값 있을 경우 Loop Break
        null_sum=0
        
        for i in textbook_check.isnull().sum():
            null_sum+=i
        if null_sum != 0 : 
            print("*********null contents in resource table")
            break
        else : pass   
        
        # story table 중복 및 Null 값 확인 
        with open('./data/story_table_text_new.pickle', 'rb') as f:story_check= pickle.load(f)
        len_dup_srid=len(story_check[story_check.duplicated(['SR ID'])])
        len_dup_text=len(story_check[story_check.duplicated(['Text'])])
        
        # 중복 있을 경우 Loop Break
        if len_dup_srid + len_dup_text !=0 :
            print("*********SR ID Duplicated : " + str(len_dup_srid))
            print("*********Text Duplicated : " + str(len_dup_text))
            break
        else : pass   
        
        # Null 값 있을 경우 Loop Break
        null_sum=0
        for i in story_check.isnull().sum():
            null_sum+=i
        if null_sum != 0 : 
            print("null contents in story table") 
            break
        else : pass   



def update_sentence_ids(*args):
    for class_name, resource_information in args:
        
        if check_duplicate(resource_information)  == False :
            break
        else: pass

        #sentence table & XT & UL ID 업데이트
        get_sentence_table_updated(class_name, resource_information)

        # XT & UL ID 중복 확인  
        with open('./data/XTID.pickle', 'rb') as f :
            token_xpos_check=pickle.load(f)
        with open('./data/ULID.pickle', 'rb') as f :
            lemma_upos_check=pickle.load(f)

        len_dup_token_xpos=len(token_xpos_check[token_xpos_check.duplicated(['XPOS:Token'])])
        len_dup_lemma_upos=len(lemma_upos_check[lemma_upos_check.duplicated(['UPOS:Lemma'])])

        # 중복 있을 경우 Loop Break
        if len_dup_token_xpos!=0 :
            print("***********XPOS Token duplicated : " + str(len_dup))
            break
        else: pass

        if len_dup_lemma_upos!=0 :
            print("***********UPOS Lemma duplicated : " + str(len_dup))
            break
        else: pass
