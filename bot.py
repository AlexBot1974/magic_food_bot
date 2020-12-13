# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:08:13 2020

@author: User
"""
import os
import pickle  #for saving trained model
import numpy as np
import pandas as pd
PATH_TO_DATA='d:/magic_food_bot/'
import telebot
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from random import sample
#token_to_id = {}
tokens=[]

def to_matrix(data, token_to_id, max_len=None, dtype='int32', batch_first = True):
    """Casts a list of names into rnn-digestable matrix"""
    
    max_len = max_len or max(map(len, data))
    data_ix = np.zeros([len(data), max_len], dtype) + token_to_id['START']

    for i in range(len(data)):
        line_ix = [token_to_id[c] for c in data[i]]
        data_ix[i, :len(line_ix)] = line_ix
        
    if not batch_first: # convert [batch, time] into [time, batch]
        data_ix = np.transpose(data_ix)

    return data_ix

class CharRNNCell(nn.Module):
    """
    Implement the scheme above as torch module
    """
    def __init__(self, num_tokens=len(tokens), embedding_size=16, rnn_num_units=64):
        super(self.__class__,self).__init__()
        self.num_units = rnn_num_units
        
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.rnn_update = nn.Linear(embedding_size + rnn_num_units, rnn_num_units)
        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)
        
    def forward(self, x, h_prev):
        """
        This method computes h_next(x, h_prev) and log P(x_next | h_next)
        We'll call it repeatedly to produce the whole sequence.
        
        :param x: batch of character ids, variable containing vector of int64
        :param h_prev: previous rnn hidden states, variable containing matrix [batch, rnn_num_units] of float32
        """
        # get vector embedding of x
        x_emb = self.embedding(x)
        
        # compute next hidden state using self.rnn_update
        x_and_h = torch.cat([x_emb, h_prev], dim=1) #YOUR CODE HERE
        h_next = self.rnn_update(x_and_h) #YOUR CODE HERE
        
        h_next = F.tanh(h_next)
        
        assert h_next.size() == h_prev.size()
        
        #compute logits for next character probs
        logits = self.rnn_to_logits(h_next)
        
        return h_next, F.log_softmax(logits, -1)
    
    def initial_state(self, batch_size):
        """ return rnn state before it processes first input (aka h0) """
        return Variable(torch.zeros(batch_size, self.num_units))
filename=PATH_TO_DATA+'finalized_char_rnn.pickle'
char_rnn = pickle.load(open(filename, 'rb'))    

#char_rnn = CharRNNCell() !!!
opt = torch.optim.Adam(char_rnn.parameters())
history = []

# load the model from disk
#names=['START','onion']
filename1=PATH_TO_DATA+'finalized_token_to_id.pickle'
token_to_id = pickle.load(open(filename1, 'rb'))
#print(token_to_id['START'])

tokens=list(token_to_id)

bot=telebot.TeleBot('TOKEN_BOT_HERE')

@bot.message_handler(content_types=["text"])
def repeat_all_messages(message): # Название функции не играет никакой роли, в принципе
    token1=message.text
    if token1 in set(list(token_to_id)):
        pass
    else:
        token1='onion'
    names=['START']
    names.append(token1)
    #bot.send_message(message.chat.id, message.text)
    #seed_phrase=['START','milk']
    seed_phrase=names
    MAX_LENGTH=9
    #max_length=MAX_LENGTH  MAX_LENGTH = max(map(len, names))
    max_length=MAX_LENGTH
    temperature=1.0
    x_sequence = [token_to_id[token] for token in seed_phrase]
    x_sequence = torch.tensor([x_sequence], dtype=torch.int64)
    hid_state = char_rnn.initial_state(batch_size=1)
    
    #feed the seed phrase, if any
    for i in range(len(seed_phrase) - 1):
        hid_state, _ = char_rnn(x_sequence[:, i], hid_state)
    
    #start generating
    for _ in range(max_length - len(seed_phrase)):
        hid_state, logp_next = char_rnn(x_sequence[:, -1], hid_state)
        p_next = F.softmax(logp_next / temperature, dim=-1).data.numpy()[0]
        
        # sample next token and push it back into x_sequence
        #next_ix = np.random.choice(len(tokens), p=p_next)
        next_ix = np.random.choice(len(tokens))
        next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
        x_sequence = torch.cat([x_sequence, next_ix], dim=1)
        
    recipe_txt='*'.join([tokens[ix] for ix in x_sequence.data.numpy()[0]])
    print('work')


    bot.send_message(message.chat.id, recipe_txt)
if __name__ == '__main__':
    bot.polling(none_stop=True)
