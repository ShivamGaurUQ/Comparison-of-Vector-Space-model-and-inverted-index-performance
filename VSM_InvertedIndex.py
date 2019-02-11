
# coding: utf-8

# In[ ]:


import glob			#python library to read open multiple files and read data from them
import re			#python library to comprehend regular expression
import os			#python library used to delete a file
from collections import Counter			#python library used to count tokens
from collections import defaultdict
import sys
import pickle
import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time




class vector_space_model:
    
    unique_tokens=[]
    index_address=""
    words=[]
    idf_index={}
    vec_matrix=np.zeros((1401,1000))
    task_3=[]
    
 
    
  
    
        
    def __init__(self):		#class constructor to read address of dataset and common words in the file
        
        self.all_tokens=[]
        self.tokens=[]
        self.stop_words=[]
        self.tokens_wo_stopwords=[]
        self.unique_tokens=[]
        self.inverted_index={}
        self.inner_index={}
        self.keywords=[]
        self.doc_list=[]
        self.term_search={}
        
        
                  
        self.prereq_inverted_index()
        
        
            
    def prereq_inverted_index(self):
        
        directory_name=input("Enter corpus directory path ")
 
        if os.path.isdir(directory_name):
            print("Directory found. Creating vector space model and inverted index")
            self.data_address=directory_name+'\cranfieldDocs\cranfield*'  										#also used to read address of file which stores data temporarily from all files in the dataset      
            self.address_com_words=directory_name+'\common_words.txt'
            vector_space_model.index_address=directory_name+"\inverted_index.pickle"
            self.process_data()
        else:
            print("Wrong directory address!")
            
            
    
            
    def process_data(self):										#function to read data from all the files in the dataset and copy all data into one duplicate file temporarily
        data_files=sorted(glob.glob(self.data_address))		#uses glob library to open all files one by one
        for data_file in data_files:
            info=open(data_file,"r")
            if info.mode=='r':
                contents=info.read()
                sgml_tags = re.sub('<[^<]+>', "", contents)
                info.close()
                with open(data_file, "w") as remove_tag:
                    remove_tag.write(sgml_tags)
                    remove_tag.close()
                    
                    
                self.stop_words.clear()
                self.tokens_wo_stopwords.clear()					#the function also removes common words from the tokens obtained in task 1
                with open(self.address_com_words) as com_words_file:
                    for sentence in com_words_file:
                        for com_word in re.findall(r'\w+', sentence):
                            self.stop_words.append(com_word.lower())
                
                
                self.tokens.clear()
                with open(data_file,"r") as doc_file:		
                    for sentence in doc_file:
                        for alpha_num_token in re.findall(r'\w+', sentence):		#Uses 're' library to identify alphanumeric tokens
                            self.tokens.append(alpha_num_token.lower())
                    doc_file.close()
                
                self.tokens_wo_stopwords=self.tokens.copy()
                for com_word in list(self.tokens_wo_stopwords):
                    if com_word in self.stop_words:
                        self.tokens_wo_stopwords.remove(com_word)
                
                
                with open(data_file, "w") as write_token:
                    for each_token in self.tokens_wo_stopwords:
                        self.all_tokens.append(each_token)
                        write_token.write(each_token)
                        write_token.write("\n")
                    write_token.close()
            
        self.unique_tokens=set(list(self.all_tokens))
        
        self.create_inverted_index()
            
    def create_inverted_index(self):
        
        data_files=sorted(glob.glob(self.data_address))		#uses glob library to open all files one by one
        
        temp_list=[]
        doc_num=0
        
        for data_file in data_files:
            
            self.inner_index.clear()
            info=open(data_file,"r")
            if info.mode=='r':
                doc_num=doc_num+1                
                for newline in info:
                    newline=newline.rstrip()
                    wordlist=newline.split()
                    
                    for word in wordlist:
                        if word in self.unique_tokens:                                
                            self.inner_index[word]=self.inner_index.get(word,0) + 1
                            
               
                
                for word, count in self.inner_index.items():
                    if word in self.inverted_index:
                       
                        temp_list=self.inverted_index[word]
                       
                        tple=(doc_num,count)
                        temp_list.append(tple)
                        self.inverted_index[word]=temp_list                      
                        
                                                
                        
                        
                    else:
                        
                        tple=(doc_num,count)
                        self.inverted_index[word]=list(tple)
                                     
               
            
                
                
            
            
            
            info.close()
        print('Inverted index created. Writing index to file.')
        write_index_to_file=open(vector_space_model.index_address,"wb")
        pickle.dump(self.inverted_index,write_index_to_file)
        write_index_to_file.close()
        
        vector_space_model.unique_tokens=self.unique_tokens.copy()
        self.term_weight()
        
        
        
        
        
    def term_weight(self):
        most_common_terms=Counter(self.all_tokens).most_common(1000)
        
        self.index={}
        read_inverted_index=open(vector_space_model.index_address,"rb")
        self.index=pickle.load(read_inverted_index)
        
        i=0
        
        doc_list=[]
        num_of_doc=0
        for each_term in most_common_terms:
            word,freq=each_term
            vector_space_model.words.append(word)           
                
        
        for each_word in vector_space_model.words:
            
            doc_list=self.index[each_word].copy()
            if each_word in self.index:
                
                vector_space_model.vec_matrix[doc_list[0]][i]=1+math.log(doc_list[1])
                num_of_doc=num_of_doc+1
                var=doc_list.pop(0)
                var=doc_list.pop(0)
                for each_indice in doc_list:
                        doc,count=each_indice
                        vector_space_model.vec_matrix[doc][i]=1+math.log(count)
                        num_of_doc=num_of_doc+1
            
            
                vector_space_model.idf_index[each_word]=math.log(1401/num_of_doc)
            doc_list=self.index[each_word].copy()
            if each_word in self.index:
                
                vector_space_model.vec_matrix[doc_list[0]][i]=vector_space_model.vec_matrix[doc_list[0]][i]*math.log(1401/num_of_doc)
                
                var=doc_list.pop(0)
                var=doc_list.pop(0)
                for each_indice in doc_list:
                        doc,count=each_indice
                        vector_space_model.vec_matrix[doc][i]=vector_space_model.vec_matrix[doc][i]*math.log(1401/num_of_doc)
                        
            
            num_of_doc=0
            i=i+1                
               
        
        i=1
        j=0
        
        while i<=1400:
            sumrow=0
            j=0
            root=0
            while j<1000:
                sumrow=sumrow+(vector_space_model.vec_matrix[i][j]*vector_space_model.vec_matrix[i][j])
                j=j+1
            root=sumrow**0.5
            j=0
            while j<1000:
                vector_space_model.vec_matrix[i][j]=vector_space_model.vec_matrix[i][j]/root
                j=j+1
            i=i+1
            if(i>1400):
                break

        
        
    
    def search(self):
        self.keywords.clear()
        self.doc_list.clear()
        num_of_keywords=input("Enter number of keywords for search (max 3)")
        
        i=0
        self.term_search.clear()
        while i<int(num_of_keywords):
            keyword=input('Enter keyword ')
            self.keywords.append(keyword)
            i=i+1
            
        
        #ranking documents using vector space model only (Task:2)
        
        query_matrix=np.zeros((1000))
        
        i=0
        
        for each_word in vector_space_model.words:
            if self.keywords.count(each_word)>0:
                
                
                query_matrix[i]=(1+math.log(self.keywords.count(each_word)))*vector_space_model.idf_index[each_word]
                
            else:
                
                query_matrix[i]=0.0
            i=i+1
                
    
           
        sumrow=0
        j=0
        root=0
        while j<1000:
            sumrow=sumrow+(query_matrix[j]*query_matrix[j])
            j=j+1
        root=sumrow**0.5
        k=0
        while k<1000:
            query_matrix[k]=query_matrix[k]/root
            k=k+1
        
        similarity=np.zeros((1401))
        i=1
        top_ten_docs={}
        while i<=1400:
            
            
            similarity[i]=np.matmul(vector_space_model.vec_matrix[i],query_matrix)
            top_ten_docs[i]=similarity[i]
            i=i+1
            
        slist=similarity.tolist()
        
        print(sorted(slist, reverse=True)[:10])
        
        print(sorted(top_ten_docs.items(), key=lambda x : x[1], reverse=True)[:10])
        
        task3_obj=task3(self.keywords)
        self.inv_doc_list=[]
        for tupl in vector_space_model.task_3:
            doc,count=tupl
            self.inv_doc_list.append(doc)
            
        
            
            
            
            
            
        similaritytsk3=np.zeros((1401))
        
        top_ten_docs.clear()
        for i in self.inv_doc_list:
            
            
            similaritytsk3[i]=np.matmul(vector_space_model.vec_matrix[i],query_matrix)
            top_ten_docs[i]=similaritytsk3[i]
            
            
        tsk3list=similaritytsk3.tolist()
        print("task3")
        print(sorted(tsk3list, reverse=True)[:10])
        
        print(sorted(top_ten_docs.items(), key=lambda x : x[1], reverse=True)[:10])
        
        
            
    def bonus_vsm(self):
        loop=0
        start = time.time()
        while loop<10:
            if loop==10:
                break
            
            self.doc_list.clear()
        
            self.term_search.clear()
        
        
        #ranking documents using vector space model only (Task:2)
        
            query_matrix=np.zeros((1000))
        
            i=0
        
            for each_word in vector_space_model.words:
                if self.keywords.count(each_word)>0:
                
                
                    query_matrix[i]=(1+math.log(self.keywords.count(each_word)))*vector_space_model.idf_index[each_word]
                
                else:
                
                    query_matrix[i]=0.0
                i=i+1
                
    
           
            sumrow=0
            j=0
            root=0
            while j<1000:
                sumrow=sumrow+(query_matrix[j]*query_matrix[j])
                j=j+1
            root=sumrow**0.5
            k=0
            while k<1000:
                query_matrix[k]=query_matrix[k]/root
                k=k+1
        
            similarity=np.zeros((1401))
            i=1
            top_ten_docs={}
            while i<=1400:
            
           
                similarity[i]=np.matmul(vector_space_model.vec_matrix[i],query_matrix)
                top_ten_docs[i]=similarity[i]
                i=i+1
            
            slist=similarity.tolist()       
            loop=loop+1
            end = time.time()
            return end-start
        
    
    
    
    def bonus_invertedindex(self):
        loop=0
        start = time.time()
        while loop<10:
            if loop==10:
                break
            
            self.doc_list.clear()
        
            self.term_search.clear()
        
        
        #ranking documents using vector space model and inverted index  (Task:3)
        
            query_matrix=np.zeros((1000))
        
            i=0
        
            for each_word in vector_space_model.words:
                if self.keywords.count(each_word)>0:
                
                
                    query_matrix[i]=(1+math.log(self.keywords.count(each_word)))*vector_space_model.idf_index[each_word]
                
                else:
                
                    query_matrix[i]=0.0
                i=i+1
                
    
           
            sumrow=0
            j=0
            root=0
            while j<1000:
                sumrow=sumrow+(query_matrix[j]*query_matrix[j])
                j=j+1
            root=sumrow**0.5
            k=0
            while k<1000:
                query_matrix[k]=query_matrix[k]/root
                k=k+1
        
            
            
            
            
            similarity=np.zeros((1401))
            
            top_ten_docs={}
            
            for i in self.inv_doc_list:
            
            
                similarity[i]=np.matmul(vector_space_model.vec_matrix[i],query_matrix)
                top_ten_docs[i]=similarity[i]
                i=i+1
            
            slist=similarity.tolist() 
            
            
            loop=loop+1
            end = time.time()
            return end-start
        
    
    
    
    
    
    
    
    
        
class task3:
        
    def __init__(self,query):
        self.keywords=query.copy()
        self.doc_list=[]
        self.index={}
        read_inverted_index=open(vector_space_model.index_address,"rb")
        self.index=pickle.load(read_inverted_index)
        self.term_search={}
        self.search_index()
    
    def search_index(self):               
        
            
        for term in self.keywords:
            
            if term in vector_space_model.unique_tokens:
                
                self.doc_list=self.index[term]
                
           
                self.term_search[self.doc_list[0]]=self.term_search.get(self.doc_list[0],0)+self.doc_list[1]
           
                var=self.doc_list.pop(0)
                var=self.doc_list.pop(0)
            
            
            
            
            
            
                for doc_freq in self.doc_list:
                    doc,frequency=doc_freq
                
                    self.term_search[doc]=self.term_search.get(doc,0)+frequency
                    
                
        
        sorted_docs=sorted(self.term_search.items(), key=lambda x : x[1], reverse=True)
        if len(sorted_docs)>0:
            
            vector_space_model.task_3=sorted_docs.copy()
        else:
            return  NULL
        
        
        
        
        
        

        
        
if __name__=='__main__':
    search_query=vector_space_model()
    while(True):
        user_choice=input("Press 1 to search query or 2 to exit")
        if int(user_choice)==1:
            search_query.search()
            print("VSM average time taken for 10 executions")
            print(search_query.bonus_vsm()/10)
            print("VSM with inverted index average time taken for 10 executions")
            print(search_query.bonus_invertedindex()/10)
        else:
            break
        if int(user_choice)==2:
            break
        
    
        

