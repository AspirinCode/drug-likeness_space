# This code was adapted from LigGPT https://github.com/devalab/molgpt
# with modifications.


"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1 
    resid_pdrop = 0.1 
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd) #256, 256
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = 1
        
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #transpose takes as arg which dims to transpose; here, -1 beco -2
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        
        attn_save = att
        att = self.attn_drop(att)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        
        return y, attn_save

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        
        y, attn = self.attn(self.ln1(x))
        
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        print('block size ',config.block_size)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        if config.num_props:
            if config.fingerprint:
                self.prop_nn = nn.Linear(config.fingerprint, config.n_embd)
            else:
                self.prop_nn = nn.Linear(config.n_embd, config.n_embd)
            
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        
        if config.lstm:
            self.lstm = nn.LSTM(input_size = config.n_embd, hidden_size = config.n_embd, num_layers = config.lstm_layers, dropout = 0.3, bidirectional = False)
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        
        self.count=0
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    
                    no_decay.add(fpn)

        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, prop = None, scaffold = None):
        
        b, t = idx.size()
        
        self.count+=1
        
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        #if self.config.num_props:
        
        token_embeddings = self.tok_emb(idx) 
        
        position_embeddings = self.pos_emb[:, :t, :] 
        
        # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device))
        
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        
        if self.config.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            
            p = self.prop_nn(prop)
            
            p += type_embd
            
            x = torch.cat([p, x], 1)
            
        if self.config.scaffold:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            
            scaffold_embeds = self.tok_emb(scaffold)     # .mean(1, keepdim = True)
            
            if self.config.lstm:        
                scaffold_embeds = self.lstm(scaffold_embeds.permute(1,0,2))[0]
                
                scaffold_embeds = scaffold_embeds.mean(0, keepdim = True).permute(1,0,2)
                
            scaffold_embeds += type_embd
            
            x = torch.cat([scaffold_embeds, x], 1)
            
        attn_maps = []
        
        for layer in self.blocks:
            
            x, attn = layer(x)
            
            attn_maps.append(attn)
        
        x = self.ln_f(x)
       
        logits = self.head(x)
        
        if self.config.num_props or self.config.scaffold:
            num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen)      
            
            num = 1
            logits = logits[:, num:, :]
            
        loss = None
        
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, attn_maps 






