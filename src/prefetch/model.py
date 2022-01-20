import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding( nn.Module ):
    def __init__(self,  max_seq_len, embed_dim  ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        pe = torch.zeros( 1, max_seq_len,  embed_dim )
        for pos in range( max_seq_len ):
            for i in range( 0, embed_dim, 2 ):
                pe[ 0, pos, i ] = math.sin( pos / ( 10000 ** ( i/embed_dim ) )  )
                if i+1 < embed_dim:
                    pe[ 0, pos, i+1 ] = math.cos( pos / ( 10000** ( i/embed_dim ) ) )
        self.register_buffer( "pe", pe )
        ## register_buffer can register some variables that can be saved and loaded by state_dict, but not trainable since not accessible by model.parameters()
    def forward( self, x ):
        return x + self.pe[ :, : x.size(1), :]

class AddMask( nn.Module ):
    def __init__( self ):
        super().__init__()
    def forward( self, x, pad_index ):
        # here x is a batch of input sequences (not embeddings) with the shape of [ batch_size, seq_len]
        mask = x == pad_index
        return mask

class MultiHeadAttention( nn.Module ):
    def __init__(self, embed_dim, num_heads ):
        super().__init__()
        dim_per_head = int( embed_dim/num_heads )
        
        self.ln_q = nn.Linear( embed_dim, num_heads * dim_per_head )
        self.ln_k = nn.Linear( embed_dim, num_heads * dim_per_head )
        self.ln_v = nn.Linear( embed_dim, num_heads * dim_per_head )

        self.ln_out = nn.Linear(  num_heads * dim_per_head , embed_dim )

        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
    
    def forward( self, q,k,v, mask = None):
        q = self.ln_q( q )
        k = self.ln_k( k )
        v = self.ln_v( v )

        q = q.view( q.size(0), q.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )
        k = k.view( k.size(0), k.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )
        v = v.view( v.size(0), v.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )

        a = self.scaled_dot_product_attention( q,k, mask )
        new_v = a.matmul(v)
        new_v = new_v.transpose( 1,2 ).contiguous()
        new_v = new_v.view( new_v.size(0), new_v.size(1), -1 )
        new_v = self.ln_out( new_v )
        return new_v

    def scaled_dot_product_attention( self, q, k, mask = None ):
        ## note the here q and k have converted into multi-head mode 
        ## q's shape is [ Batchsize, num_heads, seq_len_q, dim_per_head ]
        ## k's shape is [ Batchsize, num_heads, seq_len_k, dim_per_head ]
        # scaled dot product
        a = q.matmul( k.transpose( 2,3 ) )/ math.sqrt( q.size(-1) )
        # apply mask (either padding mask or seqeunce mask)
        if mask is not None:
            a = a.masked_fill( mask.unsqueeze(1).unsqueeze(1) , -1e9 )  ## The newly added dimension is the multihead dimension
        # apply softmax, to get the likelihood as attention matrix
        a = F.softmax( a, dim=-1 )
        return a

class FeedForward( nn.Module ):
    def __init__( self, embed_dim, hidden_dim ):
        super().__init__()
        self.ln1 = nn.Linear( embed_dim, hidden_dim )
        self.ln2 = nn.Linear( hidden_dim, embed_dim )
    def forward(  self, x):
        net = F.relu(self.ln1(x))
        out = self.ln2(net)
        return out
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim ):
        super().__init__()
        self.mha = MultiHeadAttention( embed_dim, num_heads  )
        self.norm1 = nn.LayerNorm( embed_dim )
        self.feed_forward = FeedForward( embed_dim, hidden_dim )
        self.norm2 = nn.LayerNorm( embed_dim )
    def forward( self, x, mask, dropout_rate = 0. ):
        short_cut = x

        net = F.dropout(self.mha( x,x,x, mask ), p = dropout_rate)
        net = self.norm1( short_cut + net )
        short_cut = net
        net = F.dropout(self.feed_forward( net ), p = dropout_rate)
        net = self.norm2( short_cut + net )
        return net
   
        
class MultiHeadPoolingLayer( nn.Module ):
    def __init__( self, embed_dim, num_heads ):
        super().__init__()

        self.num_heads = num_heads
        self.dim_per_head = int( embed_dim/num_heads )
        self.ln_attention_score = nn.Linear( embed_dim, num_heads )
        self.ln_value = nn.Linear( embed_dim,  num_heads * self.dim_per_head )
        self.ln_out = nn.Linear(   num_heads * self.dim_per_head, embed_dim )
    def forward(self, input_embedding , mask=None, return_attention = False):
        a = self.ln_attention_score( input_embedding )
        v = self.ln_value( input_embedding )
        
        a = a.view( a.size(0), a.size(1), self.num_heads, 1 ).transpose(1,2)
        v = v.view( v.size(0), v.size(1),  self.num_heads, self.dim_per_head  ).transpose(1,2)
        a = a.transpose(2,3)
        if mask is not None:
            a = a.masked_fill( mask.unsqueeze(1).unsqueeze(1) , -1e9 ) 
        a = F.softmax(a , dim = -1 )
        
        new_v = a.matmul(v)
        new_v = new_v.transpose( 1,2 ).contiguous()
        new_v = new_v.view( new_v.size(0), new_v.size(1), -1 )
        new_v = F.relu(new_v)  ## update: add a linear activation here
        new_v = self.ln_out( new_v ).squeeze(1)
        if return_attention:
            return new_v, a
        else:
            return new_v


class PreEncoding( nn.Module):
    def __init__(self,  vocab_size, embed_dim, max_seq_len , pad_index ,pretrained_word_embedding  ):
        super().__init__()
        self.mask_generator = AddMask()
        if pretrained_word_embedding is not None:
            self.register_buffer( "word_embedding", torch.from_numpy(pretrained_word_embedding) )
        else:
            self.register_buffer( "word_embedding", torch.randn( vocab_size, embed_dim ) )
        self.positional_encoding = PositionalEncoding( max_seq_len, embed_dim )

        self.pad_index = pad_index

    def forward( self, input_seq ):
        mask = self.mask_generator(input_seq, self.pad_index )
        in_embed = self.positional_encoding( self.word_embedding[input_seq] )
        return in_embed, mask

class SingleParagraphEncoder( nn.Module ):
    def __init__( self, n_para_types, embed_dim, num_heads, hidden_dim, vocab_size,  max_seq_len , pad_index ,pretrained_word_embedding, num_enc_layers = 1  ):
        super().__init__()

        self.pre_encoding = PreEncoding(vocab_size, embed_dim, max_seq_len , pad_index ,pretrained_word_embedding)
        self.type_embedding = nn.Embedding( n_para_types, embed_dim )
        self.encoder_layer_list = nn.ModuleList([ TransformerEncoderLayer( embed_dim, num_heads, hidden_dim ) for _ in range( num_enc_layers )] )
        self.mha_pool = MultiHeadPoolingLayer(embed_dim, num_heads)


    def forward(self, input_seq, para_type,  dropout_rate = 0. ):
        # para_type shape: batch size x 1
        with torch.no_grad():
            net , mask = self.pre_encoding( input_seq )

        # para_type_embed = self.type_embedding( para_type )
        # net = net + para_type_embed

        for encoder_layer in self.encoder_layer_list:
            net = encoder_layer( net, mask, dropout_rate)
        para_embed = self.mha_pool( net, mask )
        return para_embed

## update: remove F.max_pool_1d since it cannot handle the mask and pytorch is buggy on it
## use multi-head pooling instead

class MultipleParagraphEncoder( nn.Module ):
    def __init__( self, n_para_types, embed_dim, num_heads , hidden_dim, max_doc_len , num_enc_layers = 1 ):
        super().__init__()
        self.positional_encoding = PositionalEncoding( max_doc_len, embed_dim )
        self.type_embedding = nn.Embedding( n_para_types, embed_dim )
        self.encoder_layer_list = nn.ModuleList([  TransformerEncoderLayer( embed_dim, num_heads, hidden_dim ) for _ in range( num_enc_layers )  ])
        self.mha_pool = MultiHeadPoolingLayer(embed_dim, num_heads)
    def forward( self, para_embed, para_type , para_mask, dropout_rate = 0. , return_attention = False  ):

        para_embed = self.positional_encoding( para_embed )
        para_type_embed = self.type_embedding( para_type )
        net = para_embed + para_type_embed

        for encoder_layer in self.encoder_layer_list:
            net = encoder_layer( net, para_mask, dropout_rate )
        out, a = self.mha_pool( net, para_mask , return_attention = True )  
        if return_attention:
            return out, a
        else:
            return out

class DocumentEncoder( nn.Module ):
    def __init__( self, embed_dim, num_heads, hidden_dim, vocab_size,  max_seq_len, max_doc_len , pad_index ,n_para_types, pretrained_word_embedding, num_enc_layers  ):
        super().__init__()
        self.single_paragraph_encoder = SingleParagraphEncoder( n_para_types, embed_dim, num_heads, hidden_dim, vocab_size,  max_seq_len , pad_index ,pretrained_word_embedding, num_enc_layers )
        self.multiple_paragraph_encoder = MultipleParagraphEncoder( n_para_types, embed_dim, num_heads , hidden_dim, max_doc_len , num_enc_layers )

    ## shape of document_paragraphs: batch_size x num_para x para_seq_len
    def forward( self, document_paragraphs, document_paragraphs_types, document_paragraphs_masks, dropout_rate = 0., return_attention = False  ):
        num_para = document_paragraphs.size(1)
        document_paragraphs_embeddings = self.single_paragraph_encoder( document_paragraphs.view(-1, document_paragraphs.size(-1)), document_paragraphs_types.view(-1).unsqueeze(1), dropout_rate )
        document_paragraphs_embeddings = document_paragraphs_embeddings.view( -1, num_para, document_paragraphs_embeddings.size(-1) )
        document_embeddings, a = self.multiple_paragraph_encoder( document_paragraphs_embeddings, document_paragraphs_types, document_paragraphs_masks, dropout_rate, return_attention = True )
        if return_attention:
            return document_embeddings, a
        else:
            return document_embeddings