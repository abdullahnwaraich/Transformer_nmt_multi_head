import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim, sqrt
import matplotlib.pyplot as plt
from typing import List
import math

class Transformer_NMT_MH(nn.Module):
   def __init__(self,src_vocab_size,tgt_vocab_size,num_src_positions,num_tgt_positions,d_model, d_internal, num_layers,num_heads):
      """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model;
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3,
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param num_heads: number of attention heads for multi-head attention
      """
      super().__init__()
      self.num_heads= num_heads
      self.d_model=d_model
      self.d_internal = d_internal
      self.src_embedding = nn.Embedding(src_vocab_size,d_model)
      self.tgt_embedding = nn.Embedding(tgt_vocab_size,d_model)
      self.src_positional_enc = PositionalEncoding(d_model,num_src_positions)
      self.tgt_positional_enc = PositionalEncoding(d_model,num_tgt_positions)
      self.tgt_out_vocab = nn.Linear(d_model,tgt_vocab_size)
      self.query_ed = nn.Linear(d_model,d_internal)
      self.key_ed = nn.Linear(d_model,d_internal)
      self.value_ed = nn.Linear(d_model,d_internal)      
      self.transformer_layers_enc = nn.ModuleList([TransformerLayerMH(d_model=d_model, d_internal=d_internal,causalMask=False,num_heads=num_heads) for _ in range(num_layers)] )
      self.transformer_layers_dec = nn.ModuleList([TransformerLayerDecoderMH(d_model=d_model, d_internal=d_internal,num_heads=num_heads) for _ in range(num_layers)] )
   def forward(self, src_indices,tgt_indices): 
    """ Take a mini-batch of source and target sentences indices, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.    
    """   
    x = self.src_embedding(src_indices) * math.sqrt(self.d_model)
    x = self.src_positional_enc(x)

    
    for layer in self.transformer_layers_enc:
        x= layer(x)
        

    y = self.tgt_embedding(tgt_indices) * math.sqrt(self.d_model)
    y = self.tgt_positional_enc(y)

    
    for layer in self.transformer_layers_dec:
        y = layer(x,y)
        

    logits = self.tgt_out_vocab(y)
    log_probs = F.log_softmax(logits, dim=-1)

    # Masking
    valid_mask = (tgt_indices.ne(0) & tgt_indices.ne(1)).unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=tgt_indices.unsqueeze(-1)).squeeze(-1)
    target_log_probs = target_log_probs * valid_mask.squeeze(-1)

    tot_target_log_probs = target_log_probs.sum(dim=1)
    tgt_words_num_to_predict = valid_mask.squeeze(-1).sum(dim=1)
    return tot_target_log_probs, tgt_words_num_to_predict

        
class TransformerLayerDecoderMH(nn.Module):
  def __init__(self, d_model, d_internal,num_heads):
      super().__init__()
      assert d_internal % num_heads == 0, "d_internal must be divisible by num_heads"
      self.num_heads = num_heads
      self.d_heads = d_internal // num_heads
      self.query_dd = nn.Linear(d_model,d_internal)
      self.value_dd = nn.Linear(d_model,d_internal)
      self.key_dd = nn.Linear(d_model,d_internal)
      self.output_linear_dd=nn.Linear(d_internal,d_model)
      self.feed_forward_dd = nn.Sequential(nn.Linear(d_model,d_model*2), nn.ReLU(inplace=False), nn.Linear(d_model*2,d_model))
      self.normalize_after_att_dd = nn.LayerNorm(d_model)
      self.normalize_after_FF_dd = nn.LayerNorm(d_model)
#      self.num_heads = num_heads
#      self.d_heads = d_internal // num_heads
      self.query_ed = nn.Linear(d_model,d_internal)
      self.key_ed = nn.Linear(d_model,d_internal)
      self.value_ed = nn.Linear(d_model,d_internal)
      self.output_linear_ed = nn.Linear(d_internal,d_model)
      self.feed_forward_ed = nn.Sequential(nn.Linear(d_model,d_model*2), nn.ReLU(inplace=False), nn.Linear(d_model*2,d_model))
      self.normalize_after_att_ed = nn.LayerNorm(d_model)
      self.normalize_after_FF_ed = nn.LayerNorm(d_model)
  def forward(self, enc_out_vecs ,output_vecs):
        """ first computes contextual decoder out then combine with enc_out_vec """
        batch_size, tgt_seq_len, _ = output_vecs.size()
        batch_size, src_seq_len, _ = enc_out_vecs.size()
        Q_star_dd = self.query_dd(output_vecs).view(batch_size, tgt_seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        K_star_dd = self.key_dd(output_vecs).view(batch_size, tgt_seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        V_star_dd = self.value_dd(output_vecs).view(batch_size, tgt_seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        cross_att_logits_dd = torch.matmul(K_star_dd, Q_star_dd.transpose(-2, -1)) / math.sqrt(self.d_heads) 
        causal_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).to(cross_att_logits_dd.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).T
        cross_att_weights_dd = F.softmax(cross_att_logits_dd+causal_mask,dim=-2)
        cross_att_out_dd= torch.matmul(V_star_dd.transpose(-2,-1),cross_att_weights_dd).transpose(-2,-1) 
        # cross_att_out.shape=(batch_size,num_heads,tgt_seq_len,d_heads)
        cross_att_out_dd=cross_att_out_dd.transpose(1,2).reshape(batch_size,tgt_seq_len,self.num_heads * self.d_heads)
        # cross_att_out.shape=(batch_size,tgt_seq_len,d_internal)
        dec_out = self.normalize_after_att_dd(self.output_linear_dd(cross_att_out_dd) + output_vecs)
        dec_out = dec_out + self.feed_forward_dd(dec_out)
        dec_out = self.normalize_after_FF_dd(dec_out)#.shape =(batch_size,tgt_seq_len,d_model)

        Q_star_ed = self.query_ed(dec_out).view(batch_size, tgt_seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        K_star_ed = self.key_ed(enc_out_vecs).view(batch_size, src_seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        V_star_ed = self.value_ed(enc_out_vecs).view(batch_size, src_seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        #K_star_ed,V_star_ed.shape =(batch_size,self.num_heads,src_seq_len,self.d_heads)
        #Q_star_ed.shape =(batch_size,self.num_heads,tgt_seq_len,self.d_heads)
        cross_att_logits_ed = torch.matmul(K_star_ed, Q_star_ed.transpose(-2, -1)) / math.sqrt(self.d_heads) 
        # cross_att_logits_ed.shape= (batch_size,self.num_heads,src_seq_len,tgt_seq_len)
        cross_att_weights_ed = F.softmax(cross_att_logits_ed,dim=-2)
        cross_att_out_ed =torch.matmul(cross_att_weights_ed.transpose(-2,-1),V_star_ed)
        #cross_att_out_ed.shape = (batch_size,self.num_heads,tgt_seq_len,self.d_heads)
        cross_att_out_ed=cross_att_out_ed.transpose(1,2).reshape(batch_size,tgt_seq_len,self.num_heads * self.d_heads)
        #cross_att_out_ed.shape = (batch_size,tgt_seq_len,d_internal)
        cross_att_out_ed = self.normalize_after_att_ed(self.output_linear_ed(cross_att_out_ed)+dec_out)
        # Apply feed-forward network with residual connection
        cross_att_out_ed = self.normalize_after_FF_ed(self.feed_forward_ed(cross_att_out_ed) + cross_att_out_ed)
        return cross_att_out_ed

class TransformerLayerMH(nn.Module):
    def __init__(self, d_model,d_internal,causalMask,num_heads):
      super().__init__()
      assert d_internal % num_heads == 0, "d_internal must be divisible by num_heads"
      self.num_heads = num_heads
      self.d_heads = d_internal // num_heads
      self.causalMask=causalMask
#      self.num_heads = num_heads
#      self.d_heads = d_internal // num_heads
      self.query = nn.Linear(d_model,d_internal)
      self.key = nn.Linear(d_model,d_internal)
      self.value = nn.Linear(d_model,d_internal)
      self.output_linear = nn.Linear(d_internal,d_model)
      self.feed_forward = nn.Sequential(nn.Linear(d_model,d_model*2), nn.ReLU(inplace=False), nn.Linear(d_model*2,d_model))
      self.normalize_after_att = nn.LayerNorm(d_model)
      self.normalize_after_FF = nn.LayerNorm(d_model)      
    def forward(self, input_vecs):
        """ Returns the output from the encoder part of the transformer and attention weights """
        # Check the input shape and handle single-batch and multi-batch cases
        # out is contextualized input so out.shape = (batch_size,seq_len,d_model)
        batch_size, seq_len, _ = input_vecs.size()
        Q_star = self.query(input_vecs).view(batch_size, seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        K_star = self.key(input_vecs).view(batch_size, seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        V_star = self.value(input_vecs).view(batch_size, seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        #Q_star,K_star,V_star.shape =(batch_size,self.num_heads,src_seq_len,self.d_heads)
        cross_att_logits = torch.matmul(K_star, Q_star.transpose(-2, -1)) / math.sqrt(self.d_heads) #.shape=(batch_size,self.num_heads,src_seq_len,src_seq_len)
        
        if self.causalMask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(cross_att_logits.device)
            causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).T
            cross_att_weights = F.softmax(cross_att_logits+causal_mask,dim=-2)  
        else: 
          cross_att_weights = F.softmax(cross_att_logits,dim=-2)
        #cross_att_weights.shape=(batch_size,num_heads,src_seq_len,src_seq_len)
        cross_att_out = torch.matmul(V_star.transpose(-2,-1),cross_att_weights).transpose(-2,-1) 
        # cross_att_out.shape=(batch_size,num_heads,src_seq_len,d_heads)
        cross_att_out=cross_att_out.transpose(1,2).reshape(batch_size,seq_len,self.num_heads * self.d_heads)
        # cross_att_out.shape=(batch_size,src_seq_len,d_internal)
        out = self.normalize_after_att(self.output_linear(cross_att_out) + input_vecs)
        out = self.normalize_after_FF(self.feed_forward(out) + out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('positional_encoding', self._create_positional_encoding(d_model, num_positions))
    
    def _create_positional_encoding(self, d_model, num_positions):
        pe = torch.zeros(num_positions, d_model)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Sinusoidal encoding for even and odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]
