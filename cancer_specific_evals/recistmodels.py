#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:48:16 2022

@author: nomanashraf
"""
import transformers
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import LSTM, Linear, Embedding, Conv1d, MaxPool1d, GRU, LSTMCell, GRUCell, Dropout, AdaptiveMaxPool1d
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset


from torch.nn import functional as F
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel


    
#********************************************************************#    
#********************************************************************# 

def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    return Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)


class CausalNetwork(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(CausalNetwork, self).__init__()
        self.word_dim = input_dim
        self.output_dims = output_dims
        self.causal_conv1 = CausalConv1d(in_channels = self.word_dim, out_channels = self.output_dims, kernel_size = 3, dilation = 1)
    
    def forward(self, x):

        x = x.permute(0,2,1)
        x = self.causal_conv1(x)
        x = x[:, :, :-self.causal_conv1.padding[0]]
        x = x.permute(0,2,1)
        # take batch axis away
        #x = x.squeeze(0)
        #print(x.shape)
        return x 
    



        

    
#********************************************************************#    
#********************************************************************#   
import torch
from torch import nn
import torch.nn.functional as F
import random, math, sys






def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval
    



def slice_diag(matrix, l, dv=None):
    """
    Take a batch of attention matrices for relative position encodings and slice out the relevant attentions. These
    are the length l sequences starting at the diagonal
    :param matrix:
    :return:
    """
    if dv is None:
        dv = d(matrix)

    h, w = matrix.size(-2), matrix.size(-1)

    assert w == 2 * l -1, f'(h, w)= {(h, w)}, l={l}'

    rest = matrix.size()[:-2]

    matrix = matrix.view(-1, h, w)
    b, h, w = matrix.size()

    result = matrix.view(b, -1)
    result = torch.cat([result, torch.zeros(b, l, device=dv)], dim=1)
    assert result.size() == (b, 2 * l * l), f'result.size() {result.size()}'

    result = result.view(b, l, 2*l)
    result = result[:, :, :l]

    result = result.view(*rest, h, l)
    return result


class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, emb, heads=8, mask=True):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(

            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
    
class HTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, dropout=0.0):
        super().__init__()
        
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=True, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tblocks(x)
        return x

    
embedding_size = 768 
num_heads = 8
depth = 8
vocab_size = 30522 
        
class TransformerL(nn.Module):
    def __init__(self, final_hidden_dim):

        super(TransformerL, self).__init__()
        self.final_hidden_dim = final_hidden_dim
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=self.final_hidden_dim, nhead=8)

    def forward(self, x):
        x = self.transformer_encoder(src = x)
        
        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)
        
        #x = self.transformer_encoder(src = x, is_causal=True)
        return x

class RecistModelTransformer(nn.Module):

    def __init__(self):
        super(RecistModelTransformer, self).__init__()
        
        self.final_hidden_dim = conf.HIDDEN_DIM
        self.doc_dim = conf.OUTPUT_DIM
        
        self.note_bert = BertClassifier()
        
        self.doc_GMAX = AdaptiveMaxPool1d(self.final_hidden_dim)

        if conf.BASELINE_RESULTS == False:
            #self.day_transformer = TransformerL(self.final_hidden_dim)
            self.day_transformer = HTransformer(emb = self.final_hidden_dim, heads=num_heads, depth=depth)
        
            
            self.recist_out = Linear(self.final_hidden_dim, 1)
        else:
            self.recist_out = Linear(self.final_hidden_dim + 1, 1)

        self.dropout = Dropout(conf.DROP_OUT)
            
        

    def forward(self, text, start_times, masks):

        text = text.squeeze(0)
        masks = masks.squeeze(0)
        
        start_times = start_times.squeeze(0)
        delta_times = torch.cat([torch.tensor([0.]).to(conf.DEVICE), start_times[1:] - start_times[:-1]])
        
        main = self.note_bert(text, masks)
        main = self.dropout(main)

        #print(main.shape)
        main = torch.cat([delta_times.unsqueeze(1), main], dim=1)        
        # now each document has been run through the CNN independently.
        
        # figure out unique start times.
        unique_start_times = torch.unique(start_times)

        # loop over unique start times
        day_tensor_list = []
        for start_time in unique_start_times:
            day_input = main[start_times == start_time]

            # batch axis back
            day_input = day_input.unsqueeze(0)
            day_output = self.doc_GMAX(day_input)
            day_output = torch.mean(day_output, dim=1)

            day_tensor_list.append(day_output)
        day_tensors = torch.cat(day_tensor_list)

        
        
        if conf.BASELINE_RESULTS == False:
            # batch axis back
            day_tensors = day_tensors.unsqueeze(0)
            #print(day_tensors.shape)
            main = self.day_transformer(day_tensors)
            main = self.dropout(main)
            # batch axis gone
            main = main.squeeze(0)
            out = torch.sigmoid(self.recist_out(main))
        else:
            out = torch.sigmoid(self.recist_out(day_tensors))
       
        return out
    
    
 
    
    
#********************************************************************#    
#********************************************************************#    


class RecistModel(nn.Module):

    def __init__(self):
        super(RecistModel, self).__init__()
        
        self.final_hidden_dim = conf.HIDDEN_DIM
        self.doc_dim = conf.OUTPUT_DIM
        
        self.vocab_pos = 31 + 1 #29 max number in train, 31 is max number in val and test
        self.pos_hidden_dim = 16
        
        self.note_bert = BertClassifier()
        
        self.doc_GMAX = AdaptiveMaxPool1d(self.final_hidden_dim)
        
        self.pos_embedding = Embedding(self.vocab_pos, self.pos_hidden_dim)

        if conf.BASELINE_RESULTS == False:
            #self.day_transformer = TransformerL(self.final_hidden_dim)
            self.day_transformer = HTransformer(emb = self.final_hidden_dim + self.pos_hidden_dim, heads=num_heads, depth=depth)
        
            
            self.recist_out = Linear(self.final_hidden_dim  + self.pos_hidden_dim, 1)
        else:
            self.recist_out = Linear(self.final_hidden_dim + 1, 1)

        self.dropout = Dropout(conf.DROP_OUT)
            
        

    def forward(self, text, start_times, masks):

        text = text.squeeze(0)
        masks = masks.squeeze(0)
        
        start_times = start_times.squeeze(0)
        delta_times = torch.cat([torch.tensor([0.]).to(conf.DEVICE), start_times[1:] - start_times[:-1]])
        
        main = self.note_bert(text, masks)
        main = self.dropout(main)

        #print(main.shape)
        main = torch.cat([delta_times.unsqueeze(1), main], dim=1)        
        # now each document has been run through the CNN independently.
        
        # figure out unique start times.
        unique_start_times = torch.unique(start_times)

        # loop over unique start times
        day_tensor_list = []
        for start_time in unique_start_times:
            day_input = main[start_times == start_time]

            # batch axis back
            day_input = day_input.unsqueeze(0)
            day_output = self.doc_GMAX(day_input)
            day_output = torch.mean(day_output, dim=1)

            day_tensor_list.append(day_output)
        day_tensors = torch.cat(day_tensor_list)

        ds_len = len(day_tensors)
        pos_lst = list(range(0, ds_len))
        position_emb = self.pos_embedding(torch.tensor(pos_lst).to(conf.DEVICE))
        
        main = torch.cat([day_tensors,position_emb], dim=1)
        
        if conf.BASELINE_RESULTS == False:
            # batch axis back
            #day_tensors = day_tensors.unsqueeze(0)
            #print(day_tensors.shape)
            
            
            main = main.unsqueeze(0)
            
            main = self.day_transformer(main)
            main = self.dropout(main)
            
            # batch axis gone
            main = main.squeeze(0)
            out = torch.sigmoid(self.recist_out(main))
        else:
            out = torch.sigmoid(self.recist_out(day_tensors))
       
        return out
    
    
 
    
    
#********************************************************************#    
#********************************************************************#   



class RecistModelEncodings(nn.Module):

    def __init__(self):
        super(RecistModelEncodings, self).__init__()
        
        self.final_hidden_dim = conf.HIDDEN_DIM
        self.doc_dim = conf.OUTPUT_DIM
        
        self.vocab_times = 2365 + 1
        self.time_hidden_dim = 16
        
        self.vocab_pos = 1188 + 1
        self.pos_hidden_dim = 16
        
        self.note_bert = BertClassifier()
        
        self.doc_GMAX = AdaptiveMaxPool1d(self.final_hidden_dim)
        
        self.time_embedding = Embedding(self.vocab_times, self.time_hidden_dim) 
        
        self.pos_embedding = Embedding(self.vocab_pos, self.pos_hidden_dim)

        if conf.BASELINE_RESULTS == False:
            #self.day_transformer = TransformerL(self.final_hidden_dim)
            self.day_transformer = HTransformer(emb = self.final_hidden_dim + self.time_hidden_dim + self.pos_hidden_dim, heads=num_heads, depth=depth)
        
            
            self.recist_out = Linear(self.final_hidden_dim + self.time_hidden_dim + self.pos_hidden_dim, 1)
        else:
            self.recist_out = Linear(self.final_hidden_dim + 1, 1)

        self.dropout = Dropout(conf.DROP_OUT)
            
        

    def forward(self, text, start_times, masks):

        text = text.squeeze(0)
        masks = masks.squeeze(0)
        
        start_times = start_times.squeeze(0)
        #delta_times = torch.cat([torch.tensor([0.]).to(conf.DEVICE), start_times[1:] - start_times[:-1]])
        
        main = self.note_bert(text, masks)
        main = self.dropout(main)

        #print(main.shape)
        #main = torch.cat([delta_times.unsqueeze(1), main], dim=1)        
        # now each document has been run through the CNN independently.
        
        # figure out unique start times.
        unique_start_times = torch.unique(start_times)

        # loop over unique start times
        day_tensor_list = []
        for start_time in unique_start_times:
            day_input = main[start_times == start_time]

            # batch axis back
            day_input = day_input.unsqueeze(0)
            day_output = self.doc_GMAX(day_input)
            day_output = torch.mean(day_output, dim=1)

            day_tensor_list.append(day_output)
        day_tensors = torch.cat(day_tensor_list)

        lst_seq = list(map(int, list(range(1,len(unique_start_times)+1))))
        times_emb = self.time_embedding(torch.tensor(lst_seq).to(conf.DEVICE))

        ds_len = len(day_tensors)
        pos_lst = list(range(0, ds_len))
        position_emb = self.pos_embedding(torch.tensor(pos_lst).to(conf.DEVICE))
        
        main = torch.cat([times_emb,day_tensors,position_emb], dim=1)
        
        if conf.BASELINE_RESULTS == False:
            # batch axis back
            #day_tensors = day_tensors.unsqueeze(0)
            #print(day_tensors.shape)
            
            
            main = main.unsqueeze(0)
            
            main = self.day_transformer(main)
            main = self.dropout(main)
            
            # batch axis gone
            main = main.squeeze(0)
            out = torch.sigmoid(self.recist_out(main))
        else:
            out = torch.sigmoid(self.recist_out(day_tensors))
       
        return out
    
    
 
    
    
#********************************************************************#    
#********************************************************************#   



#this model is for testing purpose only.    
class BaseLineModel(nn.Module):


    def __init__(self):
        super(BaseLineModel, self).__init__()
        
        self.final_hidden_dim = conf.HIDDEN_DIM
        self.doc_dim = conf.OUTPUT_DIM
        
        self.note_cnn = NoteCNN()
        
        self.doc_GMAX = AdaptiveMaxPool1d(self.final_hidden_dim + 1)
        
        #self.doc_GRU = GRU(self.final_hidden_dim + 1, self.doc_dim, bidirectional=False, batch_first=True)
        
        self.recist_out = Linear(self.final_hidden_dim + 1, 1)
        
        self.dropout = Dropout(conf.DROP_OUT)
            
    def forward(self, text, start_times):

        text = text.squeeze(0)

        start_times = start_times.squeeze(0)
        delta_times = torch.cat([torch.tensor([0.]).to(conf.DEVICE), start_times[1:] - start_times[:-1]])
        
        main = self.note_cnn(text)
        main = self.dropout(main)

        #print(main.shape)
        main = torch.cat([delta_times.unsqueeze(1), main], dim=1)        
        # now each document has been run through the CNN independently.
        
        # figure out unique start times.
        unique_start_times = torch.unique(start_times)

        # loop over unique start times
        day_tensor_list = []
        for start_time in unique_start_times:
            day_input = main[start_times == start_time]

            # batch axis back
            day_input = day_input.unsqueeze(0)
            day_output = self.doc_GMAX(day_input)
            day_output = torch.mean(day_output, dim=1)
            day_tensor_list.append(day_output)
        day_tensors = torch.cat(day_tensor_list)
        
        #day_tensor_list = []
        #for start_time in unique_start_times:
        #    day_input = main[start_times == start_time]
            
            # batch axis back
        #    day_input = day_input.unsqueeze(0)
        #    _, day_output = self.doc_GRU(day_input) # maxpool
        #    day_tensor_list.append(day_output.squeeze(0))
        
        #day_tensors = torch.cat(day_tensor_list)

        out = torch.sigmoid(self.recist_out(day_tensors))
        
        return out
    
    
    
#********************************************************************#    
#********************************************************************# 

#this model is for testing purpose only.    
class BaseLineModel(nn.Module):


    def __init__(self):
        super(BaseLineModel, self).__init__()
        
        self.final_hidden_dim = conf.HIDDEN_DIM
        self.doc_dim = conf.OUTPUT_DIM
        
        self.note_cnn = NoteCNN()
        
        self.doc_GMAX = AdaptiveMaxPool1d(self.final_hidden_dim + 1)
        
        #self.doc_GRU = GRU(self.final_hidden_dim + 1, self.doc_dim, bidirectional=False, batch_first=True)
        
        self.recist_out = Linear(self.final_hidden_dim + 1, 1)
        
        self.dropout = Dropout(conf.DROP_OUT)
            
    def forward(self, text, start_times):

        text = text.squeeze(0)

        start_times = start_times.squeeze(0)
        delta_times = torch.cat([torch.tensor([0.]).to(conf.DEVICE), start_times[1:] - start_times[:-1]])
        
        main = self.note_cnn(text)
        main = self.dropout(main)

        #print(main.shape)
        main = torch.cat([delta_times.unsqueeze(1), main], dim=1)        
        # now each document has been run through the CNN independently.
        
        # figure out unique start times.
        unique_start_times = torch.unique(start_times)

        # loop over unique start times
        day_tensor_list = []
        for start_time in unique_start_times:
            day_input = main[start_times == start_time]

            # batch axis back
            day_input = day_input.unsqueeze(0)
            day_output = self.doc_GMAX(day_input)
            day_output = torch.mean(day_output, dim=1)
            day_tensor_list.append(day_output)
        day_tensors = torch.cat(day_tensor_list)
        
        #day_tensor_list = []
        #for start_time in unique_start_times:
        #    day_input = main[start_times == start_time]
            
            # batch axis back
        #    day_input = day_input.unsqueeze(0)
        #    _, day_output = self.doc_GRU(day_input) # maxpool
        #    day_tensor_list.append(day_output.squeeze(0))
        
        #day_tensors = torch.cat(day_tensor_list)

        out = torch.sigmoid(self.recist_out(day_tensors))
        
        return out