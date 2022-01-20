import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from transformers import AutoModel

class Scorer( nn.Module):
    def __init__(self, bert_model_path, vocab_size ,embed_dim = 768 ):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_path)
        self.bert_model.resize_token_embeddings( vocab_size )
        self.ln_score = nn.Linear( embed_dim, 1 )

    def forward( self, inputs  ):
        ## inputs is 3-dimensional batch_size x 3 x seq_len 
        ## pair_masks shape:  batch_size x passage_pair
        ## input_ids, token_type_ids , attention_mask = inputs[:,0,:].contiguous(), inputs[:,1,:].contiguous(), inputs[:,2,:].contiguous()

        net = self.bert_model( **inputs )[0]
        ## CLS token's embedding
        net = net[ :, 0, : ].contiguous()
        score =  F.sigmoid( self.ln_score( F.relu( net )) ).squeeze(1)
        return score    