#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import defaultdict
import string

class POS_databasedmodel(object):
    def __init__(self):
        self.vocab = None
        self.tags = None
        self.punc = set(string.punctuation)
        
        self.emission_count = None
        self.tag_count = None
        
    def buildvocab(self,train_data):
        words = []
        for line in train_data:
            if line.strip():
                words.append(line.split('\t')[0])
                
        freq = defaultdict(int)
        for word in words:
            freq[word] +=1
            
        vocab = [key for key, value in freq.items() if (value > 1 and key != '\n')]
        vocab.sort()
        self.vocab = vocab
        return vocab
        
        
    def preprocess(self, test_data):
        prep= []
        for word in testwords:
            if word.strip() not in vocab:
                prep.append('--unk--')
            else:
                prep.append(word.strip())
        return  prep 

    def get_word_tag(self, line): 
        vocab = self.vocab
        if not line.split():
            word = "--n--"
            tag = "--s--"
            return word, tag
        else:
            word, tag = line.split()
            if word not in vocab: 
                # Handle unknown words
                word = '--unk--'
            return word, tag
        return None 
    
    def split_data(self,train_data,line):
        vocab = self.vocab        
        
        if not len(line.split()) == 2:
            return '--n--', '--s--'
        else:
            word,tag = line.split()
            if word in vocab:
                return word, tag
            else:
                
                return '--unk--', tag

                

    def get_counts(self,train_data):

        self.emission_count = defaultdict(int)
        self.tag_count = defaultdict(int)

        self.prev_tag = '--s--'
        for word_tag in train_data:
            word,tag = self.get_word_tag(word_tag)

            self.emission_count[(tag,word)] +=1
            self.tag_count[tag] +=1
            self.prev_tag = tag

        return self.emission_count,self.tag_count

    def train_model(self,train_data):
        self.buildvocab(train_data)
        emission_count, tag_counts = self.get_counts(train_data)   
        self.tags = sorted(tag_counts.keys()) 
        self.emission_count = emission_count

    def Predict(self,xtest):
        predicted = []
        for word in xtest:
            COUNT = 0
            POS = ''
            vocab = self.vocab
            states = self.tags
            emissioncount = self.emission_count
            if word in vocab:
                for pos in states:
                    key = (pos, word)
                    if key in emissioncount:
                        count = emissioncount[key]
                        if count > COUNT:
                            COUNT = count
                            POS = pos
            predicted.append(POS)
        return predicted

