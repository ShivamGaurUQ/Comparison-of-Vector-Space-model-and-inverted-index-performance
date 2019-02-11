
# coding: utf-8

# In[59]:


import glob
import re
import os
from collections import Counter


class text_transform:
    total_tokens=0
    total_unique_tokens=0
    first_fifty_tokens=0
    unique_tokens=[]
    tokens=[]
    most_common=[]
    stop_words=[]
    tokens_wo_stopwords=[]
    tsk1='D:\dataset\\task1.txt'
    tsk2='D:\dataset\\task2.txt'
        
    def __init__(self, data_address,duplicate_address, address_com_words):
        self.data_address=data_address  
        self.duplicate_address=duplicate_address
        self.address_com_words=address_com_words
        self.read_data()
            
    def read_data(self):
        data_files=sorted(glob.glob(self.data_address))
        duplicate_file=open(self.duplicate_address,"a+")
        for data_file in data_files:
            info=open(data_file,"r")
            if info.mode=='r':
                contents=info.read()
                duplicate_file.write("\n"+contents)
        info.close();
        duplicate_file.close();
        self.remove_sgml_tags()
        
    def remove_sgml_tags(self):
        all_text=open(self.duplicate_address).read()
        sgml_tags = re.sub('<[^<]+>', "", all_text)
        with open(self.duplicate_address, "w") as remove_tag:
            remove_tag.write(sgml_tags)
        self.tokenization()
    
    def tokenization(self):
        text_transform.tokens.clear()
        with open(self.duplicate_address) as duplicate_file:
            for sentence in duplicate_file:
                for alpha_num_token in re.findall(r'\w+', sentence):
                    text_transform.tokens.append(alpha_num_token.lower())
        os.remove(self.duplicate_address)
        self.save_task1_statistics()
        
            
        
    def save_task1_statistics(self):
        text_transform.total_tokens=len(text_transform.tokens)
        text_transform.unique_tokens=set(list(text_transform.tokens))
        text_transform.total_unique_tokens=len(text_transform.unique_tokens)
        counts=Counter(text_transform.tokens)
        print('Total tokens before removing stopwords')
        print(text_transform.total_tokens)
        print('\n')
        print('Unique tokens')
        print(text_transform.total_unique_tokens)
        text_transform.most_common=counts.most_common(50)
        print('\n')
        print('Top 50 tokens')
        print(text_transform.most_common)
        record=open(text_transform.tsk1,"w")
        record.write("Total tokens\n")
        record.write(str(text_transform.total_tokens))
        record.write("\n\nUnique tokens\n")
        record.write(str(text_transform.total_unique_tokens))
        record.write("\n\nMost frequent first 50 tokens\n")
        record.write(str(text_transform.most_common))
        record.write("\n\nAll tokens\n\n")
        record.write(str(text_transform.tokens))
        record.close()
        self.remove_stopwords()
        
        
        
        
    def remove_stopwords(self):
        text_transform.stop_words.clear()
        text_transform.tokens_wo_stopwords.clear()
        with open(self.address_com_words) as com_words_file:
            for sentence in com_words_file:
                for com_word in re.findall(r'\w+', sentence):
                    text_transform.stop_words.append(com_word.lower())
        
        
        text_transform.tokens_wo_stopwords=text_transform.tokens.copy()
        for com_word in list(text_transform.tokens_wo_stopwords):
            if com_word in text_transform.stop_words:
                text_transform.tokens_wo_stopwords.remove(com_word)
        self.save_task2_statistics()
                
                
    def save_task2_statistics(self):
        text_transform.total_tokens=len(text_transform.tokens_wo_stopwords)
        text_transform.unique_tokens=set(list(text_transform.tokens_wo_stopwords))
        text_transform.total_unique_tokens=len(text_transform.unique_tokens)
        counts=Counter(text_transform.tokens_wo_stopwords)
        print('\n\nTotal tokens after removing stopwords')
        print(text_transform.total_tokens)
        print('\n')
        print('Unique tokens')
        print(text_transform.total_unique_tokens)
        text_transform.most_common=counts.most_common(50)
        print('\n')
        print('Top 50 tokens')
        print(text_transform.most_common)
        record=open(text_transform.tsk2,"w")
        record.write("Total tokens\n")
        record.write(str(text_transform.total_tokens))
        record.write("\n\nUnique tokens\n")
        record.write(str(text_transform.total_unique_tokens))
        record.write("\n\nMost frequent first 50 tokens\n")
        record.write(str(text_transform.most_common))
        record.write("\n\nAll tokens\n\n")
        record.write(str(text_transform.tokens_wo_stopwords))
        record.close()
        
        
        
    
        
    
    
    
        
        
    
        
        
    
        
    
        
    
     


# In[ ]:


task=text_transform('D:\dataset\dataset\cranfieldDocs\cranfield*','D:\dataset\duplicate.txt','D:\dataset\dataset\common_words.txt')

