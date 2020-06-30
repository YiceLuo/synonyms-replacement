#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List
from collections import defaultdict
import string 
import nltk

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from transformers import BertForMaskedLM, AdamW

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    all_lemma=[]
    for i in wn.lemmas(lemma, pos):
        all_lemma=all_lemma+i.synset().lemma_names()
    all_lemma=set([i.replace('_',' ') for i in all_lemma if i !=lemma])
    return all_lemma

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    lemmadict=defaultdict(int)
    for i in wn.lemmas(context.lemma, context.pos):
        a=[j.count() for j in i.synset().lemmas()]
        for j in range(len(a)):
            lemmadict[i.synset().lemma_names()[j].lower()]=lemmadict[i.synset().lemma_names()[j].lower()]+a[j]
    del lemmadict[context.lemma.lower()]
    maximum = max(lemmadict, key=lemmadict.get)
    return maximum.replace('_',' ')

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = stopwords.words('english')
    overlap=defaultdict()
    ct=context.right_context+context.left_context
    ct=[i for i in ct if i not in stop_words and i not in string.punctuation] 
    for i in wn.lemmas(context.lemma, context.pos):
        if i.synset().lemma_names() != [context.lemma]:
            defi=[j for j in i.synset().examples()]
            defi.append(i.synset().definition())
            for j in i.synset().hypernyms():
                defi=defi+j.examples()
                defi.append(j.definition())
            defi=[j for i in defi for j in tokenize(i) if j not in stop_words and j not in string.punctuation]
            OV=[i in defi for i in ct]
            overlap[i]=sum(OV)/(len(OV)+1) 
    maximum = max(overlap, key=overlap.get)
    lemmadict={maximum.synset().lemma_names()[i]:maximum.synset().lemmas()[i].count() for i in range(len(maximum.synset().lemmas()))}
    del lemmadict[context.lemma]
    maximum = max(lemmadict, key=lemmadict.get)
    return maximum.replace('_', ' ')  
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        all_lemma=get_candidates(context.lemma,context.pos)
        all_lemma=[i for i in all_lemma if i in self.model.vocab]
        similarity=0
        for i in all_lemma:
            temp=self.model.similarity(context.lemma,i)
            if similarity<temp:
                similarity=temp
                closest=i       
        return closest


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    def predict(self, context : Context) -> str:
        all_lemma=get_candidates(context.lemma,context.pos)
        ct=context.left_context+['[MASK]']+context.right_context
        ct=[i for i in ct if i!='None']
        ct = "".join(x if x in string.punctuation else x+" " for x in ct) 
        input_toks = self.tokenizer.encode(ct)
        input_mat = np.array(input_toks).reshape((1,-1))
        predictions = self.model.predict(input_mat)
        best_words = np.argsort(predictions[0][0][input_toks.index(103)])[::-1]
        ranks = self.tokenizer.convert_ids_to_tokens(best_words[:3000])
        for i in ranks:
            if i in all_lemma:
                return i
        return wn_simple_lesk_predictor(context)
    
class lexDataset(Dataset):

    def __init__(self, filename, maxlen=80):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maxlen = maxlen
        self.file = [i for i in read_lexsub_xml(filename)]
    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        context=self.file[index]
        ct=['[CLS]']+context.left_context+['[MASK]']+context.right_context+['[SEP]']
        ct=[i for i in ct if i!='None']
        ct = "".join(x if x in string.punctuation else x+" " for x in ct)
        ct=self.tokenizer.tokenize(ct)
        input_toks = self.tokenizer.convert_tokens_to_ids(ct)

        if len(input_toks) < self.maxlen:
            padded_toks = input_toks + [0 for _ in range(self.maxlen - len(input_toks))]
        else:
            padded_toks = input_toks[:self.maxlen-1] + [102] 
        attn_mask= [1 if i !=0 else 0 for i in padded_toks]
        tokens_tensor = torch.tensor(padded_toks).to('cuda')
        attn_mask=torch.tensor(attn_mask).to('cuda')
        label=torch.tensor(self.tokenizer.convert_tokens_to_ids(context.lemma)).to('cuda')
        #idx=torch.tensor(input_toks.index(103)).to('cuda')
        idx=input_toks.index(103)
        return tokens_tensor, attn_mask,label,idx

class MLM(nn.Module):

    def __init__(self, freeze_bert = True,maxlen=80):
        super(MLM, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertForMaskedLM.from_pretrained('bert-base-uncased')
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.named_parameters():
                    if 'prediction' not in p[0]:
                        p[1].requires_grad = False
    def forward(self, seq, attn_masks,idx):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        
        reps = self.bert_layer(seq, attention_mask = attn_masks)[0]
        a=torch.zeros([len(idx), reps.shape[2]], dtype=torch.float).to('cuda')
        for i in range(reps.shape[0]):
            a[i]=reps[i,idx[i]]
        
        return a
def train(net, criterion, opti, train_loader, eps=20):
    net.train()
    for ep in range(eps):
        
        for it, (seq, attn_masks,labels,idx) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  

            #Obtaining the logits from the model
            h = net(seq, attn_masks,idx)
            
            #Computing loss
            loss = criterion(h, labels)

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

class myPredictor(object):

    def __init__(self,net): 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = net
    def predict(self, context : Context) -> str:
        all_lemma=get_candidates(context.lemma,context.pos)
        ct=context.left_context+['[MASK]']+context.right_context
        ct=[i for i in ct if i!='None']
        ct = "".join(x if x in string.punctuation else x+" " for x in ct) 
        input_toks = self.tokenizer.encode(ct)
        input_mat = torch.tensor(np.array(input_toks).reshape((1,-1)),dtype=torch.long).to('cuda')
        mask=torch.ones_like(input_mat).to('cuda')
        idx=torch.tensor([input_toks.index(103)]).to('cuda')
        
        predictions = self.model(input_mat,mask,idx)
        
        best_words = np.argsort(predictions[0].cpu().data.numpy())[::-1]
        ranks = self.tokenizer.convert_ids_to_tokens(best_words[:3000])
        
        for i in ranks:
            if i in all_lemma:
                return i
        return wn_simple_lesk_predictor(context)

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    train_set = lexDataset(filename = sys.argv[1])
    train_loader = DataLoader(train_set, batch_size = 20, num_workers = 0)
    net = MLM(freeze_bert = True).to('cuda')
    criterion = nn.CrossEntropyLoss()
    opti = optim.AdamW(net.parameters(), lr = 1e-5)
    train(net, criterion, opti, train_loader)
    print('net trained!')
    net.eval()
    model = myPredictor(net)

    #nltk.download('stopwords')
    #bert= BertPredictor()
    #nltk.download('wordnet')
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = model.predict(context)  
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
