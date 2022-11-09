import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from nltk.tokenize import sent_tokenize
import re


class RerankDataset(Dataset):
    def __init__(self, corpus = [], paper_database = {}, context_database = {},
                 tokenizer = None, 
                 rerank_top_K = 2000,
                 max_input_length = 512, 
                 padding = "max_length", 
                 truncation=True,
                 sep_token = "<sep>",
                 cit_token = "<cit>",
                 eos_token = "<eos>",
                 is_training = True,

                 n_document = 32,
                 max_n_positive = 1

                ):
        ## structure of the corpus
        self.corpus = corpus
        self.paper_database = paper_database
        self.context_database = context_database
        self.tokenizer = tokenizer
        self.rerank_top_K = rerank_top_K
        self.max_input_length = max_input_length
        self.padding = padding
        self.truncation = truncation
        self.sep_token = sep_token
        self.cit_token = cit_token
        self.eos_token = eos_token
        self.special_eos_token_id = self.tokenizer.convert_tokens_to_ids( self.eos_token )
        self.is_training = is_training
        self.n_document = n_document
        self.max_n_positive = max_n_positive

        self.irrelevance_level_for_positive = 0
        self.irrelevance_level_for_negative = 1
        
    def __len__(self):
        return len(self.corpus)

    def get_paper_text(self, paper_id):
        paper_info = self.paper_database.get( paper_id, { } )
        title = paper_info.get("title","")
        abstract = paper_info.get( "abstract", "" )
        return title + " " + abstract
    
    def __getitem__(self, idx):
        ## step 1: get the query information, based on local or global citation recommendation 
        ## step 2: get the candidate documents
        ## step 3: construct the input to the scorer model
        data = self.corpus[idx]

        context_id = data["context_id"]
        citing_id = self.context_database[context_id]["citing_id"]
        context_text = self.context_database[context_id]["masked_text"].replace( "TARGETCIT", self.cit_token )
        citing_text = self.get_paper_text( citing_id )

        positive_ids = data["positive_ids"]
        positive_ids_set = set( positive_ids )
        prefetched_ids = data["prefetched_ids"][:self.rerank_top_K]
        negative_ids = list(  set( prefetched_ids ) - set( positive_ids + [citing_id] )  )

        if self.is_training:
            ## sample up to max_n_positive positive ids
            positive_id_indices = np.arange( len( positive_ids ) )
            np.random.shuffle( positive_id_indices )
            candidate_id_list = [  positive_ids[i]  for i in positive_id_indices[:self.max_n_positive]   ]
            irrelevance_levels_list = [ self.irrelevance_level_for_positive ] * len( candidate_id_list )  
            for pos in  np.random.choice( len(negative_ids), self.n_document - len( candidate_id_list ) ):
                irrelevance_levels_list.append(self.irrelevance_level_for_negative)
                candidate_id_list.append(negative_ids[ pos ])
            irrelevance_levels_list = np.array(irrelevance_levels_list).astype(np.float32)
        else:
            candidate_id_list = prefetched_ids
            irrelevance_levels_list = np.array( [ self.irrelevance_level_for_positive if candidate_id in positive_ids_set  else self.irrelevance_level_for_negative for candidate_id in candidate_id_list  ] ).astype(np.float32)


        query_text_list = []
        candidate_text_list = []

        for candidate_id in candidate_id_list:
            candidate_text = self.get_paper_text( candidate_id )
            
            query_text_list.append( " ".join( citing_text.split()[:int( self.max_input_length * 0.35 ) ] ) + self.sep_token + context_text )
            candidate_text_list.append( candidate_text )


        encoded_seqs = self.tokenizer( query_text_list,candidate_text_list,  max_length = self.max_input_length , padding =  self.padding , truncation = self.truncation)
        
        for key in encoded_seqs:
            encoded_seqs[key] = np.asarray(encoded_seqs[key])
        
        encoded_seqs.update( {
                "irrelevance_levels": irrelevance_levels_list,
                "num_positive_ids": len( positive_ids )
               } )

        return encoded_seqs
