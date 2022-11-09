import random 
import math
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def PositionalEncoding(max_seq_len, d_model):
    '''
    PE_(pos, 2i)   =  sin(pos / power(10000, 2i / d_model))
    PE_(pos, 2i+1) =  cos(pos / power(10000, 2i / d_model))
    '''
    pe = torch.zeros([max_seq_len, d_model])
    position = torch.arange(max_seq_len).unsqueeze(1).repeat(1, d_model) # pos, [seq_len, d_model]
    div_value = torch.pow(10000, torch.arange(0, d_model, 2) / d_model) # power(10000, 2i / d_model)
    pe[:, 0::2] = torch.sin(position[:, 0::2] / div_value) # sin for 2i
    pe[:, 1::2] = torch.cos(position[:, 1::2] / div_value) # cos for 2i+1
    pe = pe.unsqueeze(0) # [bs(1), seq_len, d_model]
    
    return pe

def get_attn_pad_mask(key_inputs, pad_id, query_len):                   # self_attention : [bs, query_len, query_len]
    return key_inputs.eq(pad_id).unsqueeze(1).expand(-1, query_len, -1) # cross_attention : [bs, query_len, key_len]

class PoswiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForward, self).__init__()      

        self.feed_forward = nn.Sequential(nn.Linear(config.d_model, config.feed_forward_dim),
                                          nn.Dropout(config.drop_out_raito),
                                          nn.ReLU(),
                                          nn.Linear(config.feed_forward_dim, config.d_model),
                                          nn.Dropout(config.drop_out_raito))

    def forward(self, inputs):
        return self.feed_forward(inputs)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads
        assert self.d_model % self.num_att_heads == 0, "d_model({}) % num_att_heads({}) = {}. It should be 0.".format(self.d_model, self.num_att_heads, self.d_model % self.num_att_heads)
        
        self.d_head = int(self.d_model / self.num_att_heads)
        
        self.query_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.key_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)

        self.scaled_dot_attn = ScaledDotProductAttention(config, self.d_head)
        self.linear = nn.Linear(self.d_head * self.num_att_heads, self.d_model)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, query_len, d_head]
        key = self.key_proj(key).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, key_len, d_head]
        value = self.value_proj(value).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, value_len, d_head]

        if attn_mask is not None:
          attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_att_heads, 1, 1) # [bs, query_len, key_len] -> [bs, num_heads, query_len, key_len]

        context, attn_prob = self.scaled_dot_attn(query, key, value, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_att_heads * self.d_head)
        
        output = self.linear(context)
        
        return output, attn_prob

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config
        self.scale = d_head ** 0.5

    def forward(self, query, key, value, attn_mask=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale # [bs, num_heads, query_len, key_len]
        
        if attn_mask is not None:
          scores.masked_fill_(attn_mask, -1e4)
        
        attn_prob = nn.Softmax(dim=-1)(scores)
        # 성능 관련 실험 필요. 허깅페이스에서는 dropout 사용함
        # attn_prob = nn.Dropout(self.config.drop_out_raito)(attn_prob)
        context = torch.matmul(attn_prob, value) # [bs, num_heads, query_len, d_head]
                                                  
        return context, attn_prob

class LinearUnifiedNestedAttention(nn.Module):
    def __init__(self, config):
        super(LinearUnifiedNestedAttention, self).__init__()
        self.pack_attention = MultiHeadAttention(config)
        self.unpack_attention = MultiHeadAttention(config)

    def forward(
            self,
            query: torch.FloatTensor,
            key: torch.FloatTensor,
            value: torch.FloatTensor,
            p: torch.FloatTensor,
            attention_padding_mask: torch.BoolTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        Yp, _ = self.pack_attention(query=p, 
                                    key=key, 
                                    value=value, 
                                    attn_mask=attention_padding_mask)
        Yx, _ = self.unpack_attention(query=query, 
                                      key=Yp, 
                                      value=Yp, 
                                      attn_mask=None)
        return Yp, Yx

class LunaTransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(LunaTransformerEncoderLayer, self).__init__()

        self.luna_self_attention = LinearUnifiedNestedAttention(config) 
        self.feed_forward = PoswiseFeedForward(config)

        self.Yp_layer_norm = nn.LayerNorm(config.d_model)
        self.Yx_layer_norm = nn.LayerNorm(config.d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(config.d_model)


    def forward(self, inputs, p, padding_mask):
        Yp, Yx   = self.luna_self_attention(inputs, inputs, inputs, p, padding_mask)
        Yp = self.Yp_layer_norm(Yp + p)
        Yx = self.Yx_layer_norm(Yx + inputs)
        outputs = self.feed_forward(Yx)
        outputs = self.feed_forward_layer_norm(outputs + Yx)

        return outputs, Yp

class Luna_TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(Luna_TransformerEncoder, self).__init__()
        self.config = config
        self.sqrt_dim = math.sqrt(config.d_model)
        self.dropout = nn.Dropout(p=config.drop_out_raito)

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_encoding = PositionalEncoding(config.max_enc_len, config.d_model)

        self.projected_embedding_length = config.project_embedding_length
        self.projected_embeddings = nn.Parameter(torch.Tensor(self.projected_embedding_length, config.d_model))
        nn.init.normal_(self.projected_embeddings, mean=0.0, std=config.d_model ** -0.5)
        self.projected_position_embeddings = PositionalEncoding(self.projected_embedding_length, config.d_model)

        self.layers = nn.ModuleList([LunaTransformerEncoderLayer(config) for _ in range(config.num_enc_layers)])

    def forward(self, enc_inputs, padding_mask=None):

        outputs = self.word_embedding(enc_inputs) * self.sqrt_dim + self.pos_encoding.to(enc_inputs.device)
        
        projected_embedded = self.projected_embeddings * self.sqrt_dim + self.projected_position_embeddings[0].to(enc_inputs.device)
        projected_embedded = projected_embedded.expand((outputs.shape[0],) + projected_embedded.size())

        if padding_mask == None:
            padding_mask = get_attn_pad_mask(enc_inputs, self.config.pad_id, self.projected_embedding_length)
            
        else:
            padding_mask = get_attn_pad_mask(padding_mask, 0, self.projected_embedding_length)

        outputs = self.dropout(outputs)
        projected_embedded = self.dropout(projected_embedded)

        for layer in self.layers:
            outputs, projected_embedded = layer(inputs=outputs, 
                                                p=projected_embedded,
                                                padding_mask=padding_mask)
        
        return outputs, projected_embedded

















