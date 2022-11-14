import math

import torch
import torch.nn as nn
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

def get_attn_pad_mask(key_inputs, pad_id):                  
    return key_inputs.eq(pad_id)

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

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config
        self.scale = d_head ** -0.5
        

    def forward(self, query, key, value, attn_mask=None):
      
        query = query * self.scale
        scores = torch.matmul(query, key.transpose(-2, -1)) # [bs, num_heads, query_len, key_len]
        
        if attn_mask is not None:
          scores.masked_fill_(attn_mask, -1e4)
        
        attn_prob = nn.Softmax(dim=-1)(scores)
        # attn_prob = nn.Dropout(self.config.drop_out_raito)(attn_prob)
        context = torch.matmul(attn_prob, value) # [bs, num_heads, query_len, d_head]
                                                  
        return context, attn_prob

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads
        assert self.d_model % self.num_att_heads == 0, "d_model({}) % num_att_heads({}) = {}. It should be 0.".format(self.d_model, self.num_att_heads, self.d_model % self.num_att_heads)
        
        self.d_head = int(self.d_model / self.num_att_heads)
        
        self.query_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)

        if config.tie_key_value is True:
            self.key_value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        else:
            self.key_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
            self.value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
            self.key_value_proj = None

        self.scaled_dot_attn = ScaledDotProductAttention(config, self.d_head)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, query_len, d_head]
        
        if self.key_value_proj is None:
            key = self.key_proj(key).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, key_len, d_head]
            value = self.value_proj(value).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, value_len, d_head]

        else:
            key = value = self.key_value_proj(key).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, query_len, d_head] 

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) # [bs, key_len] -> [bs, 1, 1, key_len]

        context, attn_prob = self.scaled_dot_attn(query, key, value, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_att_heads * self.d_head)
        
        return context, attn_prob

class LinearUnifiedNestedAttention(nn.Module):
    def __init__(self, config):
        super(LinearUnifiedNestedAttention, self).__init__()
        self.pack_attention = MultiHeadAttention(config)
        self.unpack_attention = MultiHeadAttention(config)

        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads
        assert self.d_model % self.num_att_heads == 0, "d_model({}) % num_att_heads({}) = {}. It should be 0.".format(self.d_model, self.num_att_heads, self.d_model % self.num_att_heads)
        
        self.d_head = int(self.d_model / self.num_att_heads)

        self.linear = nn.Linear(self.d_head * self.num_att_heads, self.d_model)

    def forward(
            self,
            query,
            key,
            value,
            p,
            input_mask = None,
            p_mask = None,
    ):
        Yp, _ = self.pack_attention(query=p, 
                                    key=key, 
                                    value=value, 
                                    attn_mask=input_mask)
        Yx, _ = self.unpack_attention(query=query, 
                                      key=Yp, 
                                      value=Yp, 
                                      attn_mask=p_mask)

        Yx = self.linear(Yx)

        return Yp, Yx

class LunaTransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(LunaTransformerEncoderLayer, self).__init__()

        self.luna_self_attention = LinearUnifiedNestedAttention(config) 
        self.feed_forward = PoswiseFeedForward(config)

        self.Yp_layer_norm = nn.LayerNorm(config.d_model)
        self.Yx_layer_norm = nn.LayerNorm(config.d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(config.d_model)


    def forward(self, inputs, p, input_mask, p_mask):
        Yp, Yx   = self.luna_self_attention(inputs, inputs, inputs, p, input_mask, p_mask)
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
        self.dynamic_projection = config.dynamic_projection

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_encoding = PositionalEncoding(config.max_enc_len, config.d_model)

        self.projected_embedding_length = config.p_length
        self.projected_embeddings = nn.Parameter(torch.Tensor(self.projected_embedding_length, config.d_model))
        nn.init.normal_(self.projected_embeddings, mean=0.0, std=config.d_model ** -0.5)
        self.projected_position_embeddings = PositionalEncoding(self.projected_embedding_length, config.d_model)

        self.layers = nn.ModuleList([LunaTransformerEncoderLayer(config) for _ in range(config.num_enc_layers)])

    def forward(self, inputs, input_mask=None, input_lengths=None):

        outputs = self.word_embedding(inputs) * self.sqrt_dim + self.pos_encoding.to(inputs.device)
        
        projected_embedded = self.projected_embeddings * self.sqrt_dim + self.projected_position_embeddings[0].to(inputs.device)
        projected_embedded = projected_embedded.expand((outputs.shape[0],) + projected_embedded.size())

        if input_mask is None:
            input_mask = get_attn_pad_mask(inputs, self.config.pad_id)

        if self.dynamic_projection:
            if input_lengths is None:
              input_lengths = torch.sum(inputs.ne(self.config.pad_id), dim=-1)

            pidx = torch.arange(self.config.p_length).unsqueeze(0).to(projected_embedded.device)
            p_mask = pidx.ge(input_lengths.unsqueeze(1))
        else:
            p_mask = None
            

        outputs = self.dropout(outputs)
        projected_embedded = self.dropout(projected_embedded)

        for layer in self.layers:
            outputs, projected_embedded = layer(inputs=outputs, 
                                                p=projected_embedded,
                                                input_mask=input_mask,
                                                p_mask=p_mask)
        
        return outputs, projected_embedded, input_mask, p_mask

def efficient_causal_attention_parallel(x, y, z):
    """
    efficient causal attention operation
    Args:
        x (Tensor): Tensor with shape `(batch, n, d1)`
        y (Tensor): Tensor with shape `(batch, n, d1)`
        z (Tensor): Tensor with shape '(batch, n, d2)`
    return:
    """
    bsz, n, d1 = x.size()
    # (bsz, n, d1, 1) x (bsz, n, 1, d2) -> (bsz, n, d1, d2)
    sum_mat = torch.matmul(y.unsqueeze(3), z.unsqueeze(2))
    accum_mat = torch.cumsum(sum_mat, dim=1)
    # (bsz, n, 1, d1) x (bsz, n, d1, d2) -> (bsz, n, 1, d2) -> (bsz, n, d2)
    res = torch.matmul(x.unsqueeze(2), accum_mat).squeeze(2)
    # (1, n, 1)
    length_div = torch.arange(1, n + 1, device=x.device).unsqueeze(0).unsqueeze(2)
    res = res / length_div
    return res

class LunaCausalAttention(nn.Module):
    def __init__(self, config):
        super(LunaCausalAttention, self).__init__()

        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads= config.num_att_heads
        assert self.d_model % self.num_att_heads == 0
        self.d_head = int(self.d_model / self.num_att_heads)
        self.scale = self.d_head ** -0.5

        self.query_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.pq_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.pc_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)

        if config.tie_key_value is True:
            self.key_value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        else:
            self.key_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
            self.value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
            self.key_value_proj = None
        
        self.out_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)

    def forward(
            self,
            query,
            p,
            dec_input_mask,
            p_mask,
    ):

        query = query.transpose(0,1).contiguous()
        p = p.transpose(0,1).contiguous()

        dec_input_len, batch_size, embed_dim = query.size()

        p_len = p.shape[0]

        pq = self.pq_proj(p).view(p_len, batch_size, self.num_att_heads, self.d_head)
        pq = pq.permute(1, 2, 0, 3)
        pq = pq * self.scale

        p_context = self.pc_proj(query)
        p_context = p_context.view(dec_input_len, batch_size * self.num_att_heads, self.d_head).transpose(0, 1)

        pq = pq.view(batch_size * self.num_att_heads, -1, self.d_head).transpose(1, 2)

        pattn_weights = p_context.bmm(pq)
        pattn_weights = nn.functional.softplus(pattn_weights, beta=math.log(2.0))

        q = self.query_proj(query)
        q = q.view(dec_input_len, batch_size * self.num_att_heads, self.d_head).transpose(0, 1)
        q = q * self.scale

        if self.key_value_proj is None:
            k = self.key_proj(query)
            k = k.view(dec_input_len, batch_size * self.num_att_heads, self.d_head).transpose(0, 1)

            v = self.value_proj(query)
            v = v.view(dec_input_len, batch_size * self.num_att_heads, self.d_head).transpose(0, 1)
        else:
            k = v = self.key_value_proj(query).view(dec_input_len, batch_size * self.num_att_heads, self.d_head).transpose(0, 1)
    
        attn_weights = efficient_causal_attention_parallel(q, k, pattn_weights)

        assert list(attn_weights.size()) == [batch_size * self.num_att_heads, dec_input_len, p_len]

        if p_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_att_heads, dec_input_len, p_len)
            attn_weights = attn_weights.masked_fill(p_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 1e-9)
            attn_weights = attn_weights.view(batch_size * self.num_att_heads, dec_input_len, p_len)

        attn_probs_float = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = attn_probs_float.type_as(attn_weights)

        attn = efficient_causal_attention_parallel(attn_probs, pattn_weights, v)

        attn = attn.transpose(0, 1).contiguous().view(dec_input_len, batch_size, embed_dim)
        attn = self.out_proj(attn)
        attn = attn.transpose(0, 1)

        return attn

class LunaTransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(LunaTransformerDecoderLayer, self).__init__()
        
        self.luna_dec_self_attention = LunaCausalAttention(config)
        self.cross_attention = LinearUnifiedNestedAttention(config)
        
        self.Yp_layer_norm = nn.LayerNorm(config.d_model)
        self.Yx_layer_norm = nn.LayerNorm(config.d_model)
        
        self.feed_forward = PoswiseFeedForward(config)
        self.feed_forward_layer_norm = nn.LayerNorm(config.d_model)


    def forward(self,
                dec_outputs, 
                p,
                enc_outputs, 
                dec_input_mask, 
                p_mask,
                enc_input_mask,
                ):

        self_att_outputs  = self.luna_dec_self_attention(dec_outputs,
                                                         p,
                                                         dec_input_mask,
                                                         p_mask)
        
        dec_outputs = self_att_outputs + dec_outputs

        Yp, Yx   = self.cross_attention(dec_outputs, enc_outputs, enc_outputs, p, enc_input_mask, p_mask)

        Yp = self.Yp_layer_norm(Yp + p)
        Yx = self.Yx_layer_norm(Yx + dec_outputs)
        outputs = self.feed_forward(Yx)
        outputs = self.feed_forward_layer_norm(outputs + Yx)
                                      
        return outputs, Yp

class Luna_TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(Luna_TransformerDecoder, self).__init__()
        self.config = config
        self.sqrt_dim = math.sqrt(config.d_model)
        self.p_dropout = nn.Dropout(p=config.drop_out_raito)

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_encoding = PositionalEncoding(config.max_dec_len, config.d_model)

        if config.decoder_only is True:
          self.projected_embedding_length = config.p_length
          self.projected_embeddings = nn.Parameter(torch.Tensor(self.projected_embedding_length, config.d_model))
          nn.init.normal_(self.projected_embeddings, mean=0.0, std=config.d_model ** -0.5)
          self.projected_position_embeddings = PositionalEncoding(self.projected_embedding_length, config.d_model)

        self.layers = nn.ModuleList([LunaTransformerDecoderLayer(config) for _ in range(config.num_dec_layers)])

        self.fc = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def decoder_step(self,
                     dec_inputs,
                     enc_outputs,
                     p,
                     enc_input_mask=None, 
                     p_mask=None):

        dec_outputs = self.word_embedding(dec_inputs) * self.sqrt_dim + self.pos_encoding[:, :dec_inputs.size(1)].to(dec_inputs.device)

        if p is None:
            p = self.projected_embeddings * self.sqrt_dim + self.projected_position_embeddings[0].to(enc_inputs.device)
            p = p.expand((outputs.shape[0],) + p.size())
            if self.dynamic_projection:
              if input_lengths is None:
                input_lengths = torch.sum(inputs.ne(self.config.pad_id), dim=-1)
                pidx = torch.arange(self.config.p_length).unsqueeze(0).to(p.device)
                p_mask = pidx.ge(input_lengths.unsqueeze(1))
              
        dec_input_mask = get_attn_pad_mask(dec_inputs, self.config.pad_id)

        for layer in self.layers:
            dec_outputs, p = layer(dec_outputs, p,
                                   enc_outputs, 
                                   dec_input_mask, 
                                   p_mask,
                                   enc_input_mask)

        return dec_outputs, p

    def forward(self,
                dec_inputs,
                enc_outputs,
                p=None,
                enc_input_mask=None,
                p_mask=None):
       

        if dec_inputs is not None:
            dec_outputs, p = self.decoder_step(dec_inputs=dec_inputs,
                                               enc_outputs=enc_outputs,
                                               p=p,
                                               enc_input_mask=enc_input_mask, 
                                               p_mask=p_mask)
            
            dec_outputs = self.fc(dec_outputs)


        else:
            dec_inputs = torch.zeros([enc_outputs.size(0), self.config.max_dec_len], device=enc_outputs.device).long()
            dec_inputs = dec_inputs.fill_(self.config.pad_id)
            dec_inputs[:, 0] = self.config.bos_id

            dec_outputs = []
            for dec_idx in range(1, self.config.max_dec_len):
                dec_output, p = self.decoder_step(dec_inputs=dec_inputs[:, :dec_idx],
                                                   enc_outputs=enc_outputs,
                                                   p=p,
                                                   enc_input_mask=enc_input_mask, 
                                                   p_mask=p_mask)
                dec_output = self.fc(dec_output)                
                dec_outputs.append(dec_output[:, -1, :])                
                dec_inputs[:, dec_idx] = dec_outputs[-1].argmax(dim=-1)
                
            dec_outputs = torch.stack(dec_outputs, dim=1)


        return dec_outputs
    
class Luna_Transformer(nn.Module):
    def __init__(self, config):
        super(Luna_Transformer, self).__init__()
        self.config = config
        self.encoder = Luna_TransformerEncoder(config)
        if config.use_decoder == True:
            self.decoder = Luna_TransformerDecoder(config)
        
        self.init_weights()

    def init_weights(self):
        # Initialize weights for each layer
        self.apply(self.init_layer_weights)

    # ref huggingface
    # https://huggingface.co/transformers/v4.9.2/_modules/transformers/models/electra/modeling_electra.html#ElectraPreTrainedModel
    def init_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            module.eps = self.config.norm_eps

    def forward(self, 
                enc_inputs, 
                dec_inputs=None, 
                enc_self_attn_mask=None):
      
        enc_outputs, enc_p, enc_input_mask, enc_p_mask = self.encoder(enc_inputs, enc_self_attn_mask)
        
        if self.config.use_decoder == False:
            return  enc_outputs, enc_p
        
        dec_outputs = self.decoder(dec_inputs,
                                   enc_outputs,
                                   enc_p,
                                   enc_input_mask=enc_input_mask, 
                                   p_mask=enc_p_mask)

        return (dec_outputs,)
