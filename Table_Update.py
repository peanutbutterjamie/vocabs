#!/usr/bin/env python
# coding: utf-8

# # 필요한 라이브러리 importing

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
get_ipython().system('pip install nltk')
from nltk import sent_tokenize
get_ipython().system('pip install spacy')
import spacy 
get_ipython().system('pip install stanza')
nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma')
clear_output()


# In[256]:


stanza.download('en')
snlp = stanza.Pipeline(lang="en",processors='tokenize')
clear_output()


# In[302]:


with open('./reference_table_new.pickle', 'rb') as f:
    textbook_df2 = pickle.load(f)
with open('./story_table_text_new.pickle', 'rb') as f:
    story_df2 = pickle.load(f)
with open('./sentence_list_new.pickle', 'rb') as f:
    sentence_df2 = pickle.load(f)
with open('./STID_XTID_new.pickle', 'rb') as f :
    STID_XTID2=pickle.load(f)
with open('./STID_ULID_new.pickle', 'rb') as f :
    STID_LemmaID2=pickle.load(f)
with open('./XTID.pickle', 'rb') as f :
    token_xpos2=pickle.load(f)
with open('./ULID.pickle', 'rb') as f :
    lemma_upos2=pickle.load(f)


# #### 문장 분리 

# In[31]:


txt="“What is value to the customer?” is the most important question in business, yet is rarely asked. One reason is that managers are quite sure that they know the answer. Value is what they, in their business, define as quality. But this is almost always the wrong definition. The customer never buys a product but the satisfaction of a want. For the teenage girl, for instance, value in a shoe is high fashion. Price is a secondary consideration and durability is not value at all. For the same girl as a young mother, a few years later, what she looks for is durability, price and comfort."
doc = snlp(txt)
#doc_sents = [sentence.text for sentence in doc.sentences]
for sentence in doc.sentences:
    print(sentence.text)


# ## Referece, Story, Sentence, XT ID, UL ID 함수 정의

# #### 1. reference table update

# In[114]:


def update_ref_num (srpath):
    with open('./reference_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)
    reference_title=re.split("/", str(srpath))[-1] 
    # ref_code (ENG), series(ESC), label(4), gr(4), title(Esacalte 4), rscode (ESC-4)
    test_info=re.split('-', reference_title)
    ref_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+'-'+label
    
    # 새로운 레퍼런스 정보 업데이트용 리스트
    new_ref_info=[]

    # 새로운 Rf id 부여 
    lastid=textbook_df['RS ID'].nunique()
    refidnumber="RS" + str(lastid+1).zfill(3)

    
    # 리스트 업데이트 
    new_ref_info.append([refidnumber, ref_code, series, gr, label, rscode, title])
    
    # Df 로 변환 후 기존 테이블과 합친 후 저장 
    new_rf=pd.DataFrame(new_ref_info, columns=textbook_df.columns)
    reference_df_new=pd.concat([textbook_df, new_rf]).reset_index(drop=True)
    with open('./reference_table_new.pickle', 'wb') as f : 
        pickle.dump(reference_df_new, f)
    print("------------"+reference_title+"------------")
    print("ref table updated :"+ refidnumber)

    
def get_ref_table(class_name, ref_information):
    path="/Users/jamielee/Library/CloudStorage/Dropbox/AIEI - 코퍼스/코퍼스 리소스/리소스 TXT/TXT_"
    srpath=path+class_name+"/"+ref_information
    with open('./reference_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)
    # 레퍼런스 제목
    reference_title=re.split("/", srpath)[-1] 
    # ref_code (ENG), series(ESC), label(4), gr(4), title(Esacalte 4), rscode (ESC-4)
    test_info=re.split('-', reference_title)
    ref_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+'-'+label
    # <Reference table update>
    # 레퍼런스 테이블에 해당 레퍼런스가 존재하는지 확인
    if rscode in list(textbook_df['RS Code']) : 
        print(rscode + " is already in the reference table: "+ ''.join(textbook_df[textbook_df['RS Code']==rscode]['RS ID']))
    # 없는 경우 새로운 RS ID 를 부여하여 테이블 업데이트 
    else:
        update_ref_num(srpath)


# #### 2. Story update 
# - story 가 이미 Table 에 있는 경우
# - story 를 update 해야 되는 경우

# In[115]:


def get_story_ids(srpath, srlist, rsid, rscode, last_srid_num):
    with open('./story_table_text_new.pickle', 'rb') as f: story_df = pickle.load(f)
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

        sr_code=rscode+"-"+sr_order

        new_story_info.append([rsid, new_sr_id, sr_code, group, sr_order, unit, sr_title, sr_text])

    story_df2=pd.DataFrame(new_story_info, columns=story_df.columns)
    return story_df2


# In[116]:


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
        sr_code=rscode+"-"+sr_order
        
        indx=list(srdf2[srdf2['SR Code']==sr_code].index)
        if len(indx) >1 : 
            print("duplicated stories")
            print(srlist)
            break
        elif len(indx) == 1:
            srdf2.iloc[indx[0]]['Text']=sr_text
    with open('./story_table_text_new.pickle', 'wb') as f : pickle.dump(srdf2, f)
    print("story table text updated to : ./story_table_text_new.pickle")
    story_df2=srdf2[sr_idx[0]:sr_idx[-1]]
    return story_df2


# In[117]:


def udpate_story_table(class_name, ref_information):
    
    path="/Users/jamielee/Library/CloudStorage/Dropbox/AIEI - 코퍼스/코퍼스 리소스/리소스 TXT/TXT_"
    srpath=path+class_name+"/"+ref_information
    
    reference_title=re.split("/", srpath)[-1] 
    test_info=re.split('-', reference_title)
    ref_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+'-'+label
    
    
    story_list = sorted(os.listdir(srpath))
    if '.DS_Store' in story_list : story_list.remove('.DS_Store')

    with open('./story_table_text_new.pickle', 'rb') as f: story_df = pickle.load(f)
    with open('./reference_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)
    rsid=''.join(textbook_df[textbook_df['RS Code']==rscode]['RS ID'])
    if rsid not in set(story_df['RS ID']) : 
        #story_df= pd.read_excel('./story_table_text.xlsx', engine='openpyxl', index_col=0)
        last_srid_num=story_df['SR ID'].nunique()
        story_df2=get_story_ids(srpath, story_list, rsid, rscode, last_srid_num)
        
        story_df_new=pd.concat([story_df, story_df2]).reset_index(drop=True)
        with open('./story_table_text_new.pickle', 'wb') as f : 
            pickle.dump(story_df_new, f)
        
        print(reference_title +" : story table updated  - "+ rscode)
        print("updated stories :"+ str(len(story_df2)))
        return story_df_new
       
    elif rsid in set(story_df['RS ID']) :     
        #story_df= pd.read_excel('./story_table_text.xlsx', engine='openpyxl', index_col=0)
        sr_idx=list(story_df[story_df['RS ID']==rsid].index)
        
        print("story list already in the table : " + reference_title)
        print("story index : " + str(sr_idx))
        srdf2=story_df.copy()
        story_df2=check_stories(srpath, story_list, srdf2, rsid, rscode, sr_idx)
        return story_df2


# #### 3. sentence table update & st id-xt it / ul id & ulid, xtid table update

# In[118]:


def get_st_id(sentencedf) :
    sent_num=[]
    
    with open('./sentence_list_new.pickle', 'rb') as f: 
        orig_sent_df= pickle.load(f)
    len_orig_df= len(orig_sent_df)
    for i in range(1, len(sentencedf)+ 1) :
        sent_num.append("ST" + str(len_orig_df+i).zfill(5))
    sentencedf['ST ID'] = sent_num
    
    sentencedfnew=pd.concat([orig_sent_df, sentencedf]).reset_index(drop=True)
    
    
    with open('./sentence_list_new.pickle', 'wb') as f:
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


# In[119]:


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

# 첫 번 째 교재에서 ul id & xt id  부여하기 
def get_intial_xpos_lemma (tokenized_df):
    df_whole, xpos_df=get_token_xpos_ids(tokenized_df)
    df_whole2, lemma_df=get_lemma_upos_ids(df_whole)
    return df_whole2, xpos_df, lemma_df


def get_lemma_upos_ids(df_tokenized) :
    with open('./ULID.pickle', 'rb') as f :
        df_lemma_upos=pickle.load(f)
    list_lemma_upos=list(df_lemma_upos['UPOS:Lemma'])
    
    a= len(df_lemma_upos)
    ul_ids=[]
    
    for i in range(len(df_tokenized)) :
        
        pos_lem = df_tokenized["UPOS:Lemma"].iloc[i]
        
        if pos_lem== 'n/a' : ul_ids.append('n/a')
        
        elif pos_lem in list_lemma_upos: #list 혹은 set -> df에서 변환
            ul_ids.append(''.join(df_lemma_upos.loc[(df_lemma_upos["UPOS:Lemma"] == pos_lem)]['UL ID'].tolist()))
       
        elif pos_lem not in list_lemma_upos:
            new_id= "UL"+str(len(df_lemma_upos)+1).zfill(5)
            list_lemma_upos.append(pos_lem) 
            df_lemma_upos.loc[len(df_lemma_upos)+1] = [new_id, pos_lem]
            ul_ids.append(new_id)
        else: print("exception : " + \
                    str(pos_lem))
    df_tokenized['UL ID']=ul_ids 
    with open('./ULID.pickle', 'wb') as f:
        pickle.dump(df_lemma_upos, f, pickle.HIGHEST_PROTOCOL)   
    print("-newly added upos:lemma IDs -> "+ str(len(df_lemma_upos)-a))
    result_lemma= df_tokenized[['ST ID',"UL ID"]].replace('n/a', np.NaN).dropna(subset=['UL ID'])
    return df_tokenized, result_lemma




def get_token_xpos_ids(df_tokenized):
    with open('./XTID.pickle', 'rb') as f :
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
    
    with open('./XTID.pickle', 'wb') as f:
        pickle.dump(df_token_xpos, f, pickle.HIGHEST_PROTOCOL)
        
    print("-newly added token:xpos IDs -> "+ str(len(df_token_xpos)-a))
    return df_tokenized, df_tokenized[['ST ID','XT ID']]



def get_xpos_upos (df):
    tokenized_df=tokenize_sentence(df)
    df_whole, xpos_df=get_token_xpos_ids(tokenized_df)
    df_whole2, lemma_df=get_lemma_upos_ids(df_whole)
    return df_whole2, xpos_df, lemma_df


# In[184]:


def get_sentence_table_updated (class_name, ref_information):
    
    path="/Users/jamielee/Library/CloudStorage/Dropbox/AIEI - 코퍼스/코퍼스 리소스/리소스 TXT/TXT_"
    srpath=path+class_name+"/"+ref_information

    with open('./story_table_text_new.pickle', 'rb') as f: story_df = pickle.load(f)
    with open('./reference_table_new.pickle', 'rb') as f: textbook_df = pickle.load(f)

    reference_title=re.split("/", srpath)[-1] 
    test_info=re.split('-', reference_title)
    ref_code, series, label, gr, title=test_info[0], test_info[1], test_info[2], test_info[3], test_info[-1]
    rscode= series+'-'+label

    rsid=''.join(textbook_df[textbook_df['RS Code']==rscode]['RS ID'])
    storyindex=list(story_df[story_df['RS ID']==rsid].index)
    new_story_df=story_df[storyindex[0]:storyindex[-1]+1].reset_index(drop=True)

    #sentence 
    sentence_id_df=get_sent_df(new_story_df)

    print("------------"+reference_title+"------------")
    print("sentence table updated :" +str(len(sentence_id_df))+" sentences")

    token_lemma, xpos_df, lemma_df=get_xpos_upos(sentence_id_df)

    st_xt=token_lemma.replace('n/a', np.NaN).dropna(subset=['XT ID'])[['ST ID','XT ID']].reset_index(drop=True)
    with open('./STID_XTID_new.pickle', 'rb') as f :
        STID_XTID=pickle.load(f)

    new_xt_st=pd.concat([STID_XTID, st_xt]).reset_index(drop=True)

    with open('./STID_XTID_new.pickle', 'wb') as f:
        pickle.dump(new_xt_st, f)

    with open('./STID_ULID_new.pickle', 'rb') as f :
        STID_ULID=pickle.load(f)

    st_ul=lemma_df.reset_index(drop=True)

    new_ul_st=pd.concat([STID_ULID, st_ul]).reset_index(drop=True)

    with open('./STID_ULID_new.pickle', 'wb') as f:
        pickle.dump(new_ul_st, f)
    print("--XT ID UL ID Updated")


# # Table update 

# ### 1) reference, story, sentence 한 번에 업데이트

# In[121]:


def update_tables(*args):    
    # 스토리 경로 하나씩 처리 
    for class_name, ref_information in tqdm(args):
        get_ref_table(class_name, ref_information)

        udpate_story_table(class_name, ref_information)

        #sentence 
        get_sentence_table_updated(class_name, ref_information)


# ### 2) reference & stoty table update

# In[1]:


class_names=["TEST", "GRAD", "ELE"]
ref_path="/Users/jamielee/Library/CloudStorage/Dropbox/AIEI - 코퍼스/코퍼스 리소스/리소스 TXT/TXT_"
ref_folders=os.listdir(ref_path+class_names[1])
ref_folders.remove('.DS_Store')
ref_folders.sort()
ref_folders


# In[284]:


print(ref_folders[-2])
print(ref_folders[-1])


# In[166]:


def update_ref_stories(*args):    
    # 스토리 경로 하나씩 처리 
    for class_name, ref_information in tqdm(args):
        
        # reference table 업데이트
        get_ref_table(class_name, ref_information)
        

        # story table 업데이트
        udpate_story_table(class_name, ref_information)
        
        
        # reference table 중복 및 Null 값 확인 
        with open('./reference_table_new.pickle', 'rb') as f:textbook_check= pickle.load(f)
        
        len_dup_rsid=len(textbook_check[textbook_check.duplicated(['RS ID'])])
        len_dup_title=len(textbook_check[textbook_check.duplicated(['Title'])])
        
        # 중복 있을 경우 Loop Break, 없으면 Continue
        if len_dup_rsid + len_dup_title !=0 :
            print("*********RS ID Duplicated : " + str(len_dup_rsid))
            print("*********Title Duplicated : " + str(len_dup_title))
            break
        else: continue
            
            
        # Null 값 있을 경우 Loop Break, 없으면 Continue 
        null_sum=0
        
        for i in textbook_check.isnull().sum():
            null_sum+=i
        if null_sum != 0 : 
            print("*********null contents in reference table")
            break
        else: continue
        
        
        # story table 중복 및 Null 값 확인 
        with open('./story_table_text_new.pickle', 'rb') as f:story_check= pickle.load(f)
        len_dup_srid=len(story_check[story_check.duplicated(['SR ID'])])
        len_dup_text=len(story_check[story_check.duplicated(['Text'])])
        
        # 중복 있을 경우 Loop Break, 없으면 Continue
        if len_dup_srid + len_dup_text !=0 :
            print("*********SR ID Duplicated : " + str(len_dup_srid))
            print("*********Text Duplicated : " + str(len_dup_text))
            break
        else: continue
        
        
        # Null 값 있을 경우 Loop Break, 없으면 Continue 
        null_sum=0
        for i in story_check.isnull().sum():
            null_sum+=i
        if null_sum != 0 : 
            print("null contents in story table") 
            break
        else: 
            continue


# In[178]:


ref_folders


# In[285]:


update_ref_stories([class_names[1], ref_folders[-2]],
                  [class_names[1], ref_folders[-1]])


# In[286]:


with open('./reference_table_new.pickle', 'rb') as f:
    textbook_df2 = pickle.load(f)
textbook_df2


# In[287]:


with open('./story_table_text_new.pickle', 'rb') as f:
    story_df2 = pickle.load(f)
story_df2


# In[288]:


story_df2[story_df2.duplicated(['Text'])]


# ### 3) sentence & XT & UL ID table update

# In[182]:


def update_sentence_ids(*args):
    for class_name, ref_information in args:
        #sentence table & XT & UL ID 업데이트
        get_sentence_table_updated(class_name, ref_information)
        
        # XT & UL ID 중복 확인  
        with open('./XTID.pickle', 'rb') as f :
            token_xpos_check=pickle.load(f)
        with open('./ULID.pickle', 'rb') as f :
            lemma_upos_check=pickle.load(f)
        
        len_dup_token_xpos=len(token_xpos_check[token_xpos_check.duplicated(['XPOS:Token'])])
        len_dup_lemma_upos=len(lemma_upos_check[lemma_upos_check.duplicated(['UPOS:Lemma'])])
        
        # 중복 있을 경우 Loop Break, 없으면 Continue
        if len_dup_token_xpos!=0 :
            print("***********XPOS Token duplicated : " + str(len_dup))
            break
        else: continue
        if len_dup_lemma_upos!=0 :
            print("***********UPOS Lemma duplicated : " + str(len_dup))
            break
        else: continue


# In[289]:


update_sentence_ids([class_names[1], ref_folders[-2]],
                    [class_names[1], ref_folders[-1]])


# In[290]:


with open('./sentence_list_new.pickle', 'rb') as f:
    sentence_df2 = pickle.load(f)
sentence_df2

