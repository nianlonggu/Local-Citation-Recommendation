import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir) 

from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
try:
    import sent2vec
except:
    pass
import torch

if torch.cuda.is_available():
    from nearest_neighbor_search.modules import BFIndexIP
else:
    import faiss

import torch
import torch.nn as nn

import re
import nltk
from nltk.tokenize import  RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from functools import lru_cache
from nltk.corpus import stopwords
nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)
from nltk.tokenize import sent_tokenize


from model import DocumentEncoder as DocumentEncoderKernel
from datautils import Vocab as DocumentEncoderVocab
from datautils import PrefetchDataset as DocumentEncoderPrefetchDataset


class Ranker:
    """
      Note: if requires_precision_conversion = False, this means the document embedding has been pre normalized and precision-converted
    """
    def __init__(self, embeddings_path, vector_dim ,gpu_list = [], precision = "float32", requires_precision_conversion = False ):
        with open( embeddings_path, "rb" ) as f:
            embeddings = pickle.load(f)
        print("embedding loaded")
        self.index_to_id_mapper = embeddings["index_to_id_mapper"]
        self.id_to_index_mapper = embeddings["id_to_index_mapper"]

        ## first normalize the embedding before converting precision, or the precision is float32
        if requires_precision_conversion or precision == "float32": 
            self.doc_embeddings = self.normalize_embeddings( embeddings["embedding"] )
        else:
            self.doc_embeddings = embeddings["embedding"]
        
        if torch.cuda.is_available():
            self.index_ip = BFIndexIP( self.doc_embeddings, vector_dim , gpu_list, precision, requires_precision_conversion )
        else:
            assert len( gpu_list ) == 0
            self.index_ip = faiss.IndexFlatIP(vector_dim)
            self.index_ip.add( self.doc_embeddings )

        self.encoder = None

    def normalize_embeddings(self, embeddings ):
        assert len( embeddings.shape ) == 2
        normalized_embeddings = embeddings /(np.linalg.norm( embeddings, axis =1, keepdims=True )+1e-12)
        return normalized_embeddings


    def get_top_n_given_embedding( self, n, query_embedding,  indices_range = None , requires_precision_conversion = True ):
        if torch.cuda.is_available():
            top_n_indices = self.index_ip.search( query_embedding , n, indices_range, requires_precision_conversion )[1][0]
        else:
            top_n_indices = self.index_ip.search( query_embedding , int(n) )[1][0]
        return [ self.index_to_id_mapper[idx] for idx in top_n_indices ] 


    def get_top_n( self, n, query_batches_paragraphs , tokenize = True, indices_range = None,  requires_precision_conversion = True  ):
        query_embedding = self.encoder.encode( query_batches_paragraphs, tokenize )
        return self.get_top_n_given_embedding( n, query_embedding, indices_range, requires_precision_conversion )


    
class PrefetchEncoder:
    def __init__( self, model_path, unigram_words_path, embed_dim, gpu_list = [] , num_heads = 8, hidden_dim = 1024,  max_seq_len = 512, max_doc_len = 5, n_para_types = 100, num_enc_layers = 1 ):
        super().__init__()
        with open(unigram_words_path,"rb") as f:
            words = pickle.load(f)
        vocab = DocumentEncoderVocab(words)
        self.vocab = vocab
        self.dataset = DocumentEncoderPrefetchDataset( words = words, max_doc_len = max_doc_len )
        
        self.gpu_list = gpu_list
        self.n_gpu = len(gpu_list)
        self.device = torch.device( "cuda:%d"%(gpu_list[0]) if self.n_gpu >0 and torch.cuda.is_available() else "cpu"  )
        document_encoder = DocumentEncoderKernel( embed_dim, num_heads, hidden_dim, len(words),  max_seq_len, max_doc_len , vocab.pad_index ,n_para_types, pretrained_word_embedding = None, num_enc_layers = num_enc_layers )
        ckpt = torch.load( model_path, map_location="cpu" )
        document_encoder.load_state_dict( ckpt["document_encoder"] )
        self.document_encoder = document_encoder.to(self.device)
        if self.device.type == "cuda" and self.n_gpu > 1:
            self.document_encoder = nn.DataParallel( self.document_encoder, gpu_list )
    
    def encode(self, batch_paragraphs, tokenize = None ):
        ## tokenize is set to None, as in the self.dataset encode function document will be tokenized
        document_info = [] 
        for document in batch_paragraphs:
            document_info.append( self.dataset.encode_document( document ) )
        
        paragraph_seq_list, paragraph_type_list, paragraph_mask_list = list(zip(*document_info))
        paragraph_seq_list =  torch.from_numpy(np.asarray(paragraph_seq_list)).to(self.device)
        paragraph_type_list = torch.from_numpy(np.asarray( paragraph_type_list )).to(self.device)
        paragraph_mask_list = torch.from_numpy(np.asarray(paragraph_mask_list) == 1).to(self.device)
        with torch.no_grad():
            doc_embed = self.document_encoder( paragraph_seq_list, paragraph_type_list, paragraph_mask_list )
            doc_embed = doc_embed.detach().cpu().numpy()
        return doc_embed



### This is used for BM25Ranker 


class SentenceTokenizer:
    def __init__(self ):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()

        self.general_stopwords = set(stopwords.words('english')) 

    @lru_cache(100000)
    def lemmatize( self, w ):
        return self.lemmatizer.lemmatize(w)
    
    def tokenize(self, sen, remove_stopwords = False ):
        if remove_stopwords:
            sen = " ".join( [ w for w in sen.lower().split() if w not in self.general_stopwords ] )
        else:
            sen = sen.lower()
        sen = " ".join([ self.lemmatize(w) for w in self.tokenizer.tokenize( sen )   ])
        return sen


class Encoder:
    def __init__(self ):
        self.tokenizer = SentenceTokenizer()
    def tokenize_batch_paragraphs( self, batch_paragraphs  ):
        batch_paragraphs = batch_paragraphs.copy()
        for batch_idx in range(len(batch_paragraphs)):
            for para_idx in range(len( batch_paragraphs[batch_idx] )):
                batch_paragraphs[batch_idx][para_idx][0] = self.tokenizer.tokenize( batch_paragraphs[batch_idx][para_idx][0] )
        return batch_paragraphs



class Sent2vecEncoder(Encoder):
    def __init__( self, model_path ):
        super().__init__()
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(  model_path )

    def encode( self, batch_paragraphs, tokenize = True ):
        if tokenize:
            batch_paragraphs = self.tokenize_batch_paragraphs( batch_paragraphs )
        batch_text = []
        for paragraphs in batch_paragraphs:
            text = " ".join( [  para[0] for para in paragraphs ] )
            batch_text.append(text)
        return self.model.embed_sentences( batch_text )



class BM25Ranker:
    def __init__(self, inv_idx_data):
        self.tokenizer = SentenceTokenizer()
        self.inv_idx = inv_idx_data["inv_idx"]
        self.index_to_id_mapper = inv_idx_data["index_to_id_mapper"]
        self.index_to_doc_length_mapper = inv_idx_data["index_to_doc_length_mapper"]
        self.num_of_docs = inv_idx_data["num_of_docs"]
        idx_list = list(self.index_to_doc_length_mapper.keys())
        assert np.min(idx_list) == 0 and np.max(idx_list) == len(idx_list) -1
        self.doc_lengths = np.array([ self.index_to_doc_length_mapper[idx] for idx in range( len(idx_list) ) ])
        self.avg_doc_length = np.mean( self.doc_lengths )
        
    def get_scores( self, query, k = 1.2, b = 0.75, require_tokenize = True , remove_stopwords = False, ):
        if require_tokenize:
            w_list = self.tokenizer.tokenize(query, remove_stopwords = remove_stopwords).split()
        else:
            w_list = query.split()
        unique_words = {}
        for w in w_list:
            unique_words[w] = unique_words.get(w, 0) + 1
        scores = np.zeros( self.num_of_docs, dtype = np.float32 )
        for w in unique_words:
            if w not in self.inv_idx:
                continue
            Nw = len( self.inv_idx[w]["doc_indices"] )
            doc_length_w = self.doc_lengths[ self.inv_idx[w]["doc_indices"] ]
            scores[ self.inv_idx[w]["doc_indices"] ] = scores[ self.inv_idx[w]["doc_indices"] ] +  unique_words[w] * self.inv_idx[w]["term_frequencies"] *(1+k)/( self.inv_idx[w]["term_frequencies"]  + k*( 1- b + b* doc_length_w/ self.avg_doc_length )) * np.log(1+ (self.num_of_docs - Nw+0.5 )/(Nw+0.5) )    
        return scores

    def get_top_n(self, n, query, k = 1.2, b = 0.75, require_tokenize = True ,remove_stopwords = False):
        scores = self.get_scores( query, k, b, require_tokenize , remove_stopwords )
        top_n_indices = np.argsort( -scores )[:n]
        return [ self.index_to_id_mapper[idx] for idx in top_n_indices ]
