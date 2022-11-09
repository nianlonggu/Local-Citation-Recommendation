import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import re
from tqdm import tqdm
import nltk
from nltk.tokenize import  RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

class SentenceTokenizerForJaccard:
    def __init__(self ):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()

        self.general_stopwords = set(stopwords.words('english')) | set([ "<num>", "<cit>" ])

    @lru_cache(100000)
    def lemmatize( self, w ):
        return self.lemmatizer.lemmatize(w)
    
    def tokenize(self, sen, remove_stopwords = True ):
        if remove_stopwords:
            sen = " ".join( [ w for w in sen.lower().split() if w not in self.general_stopwords ] )
        else:
            sen = sen.lower()
        sen = " ".join([ self.lemmatize(w) for w in self.tokenizer.tokenize( sen )   ])
        return sen

class JaccardSim:
    def __init__(self):
        self.sent_tok = SentenceTokenizerForJaccard()

    def compute_sim( self, textA, textB ):

        textA_words = set(self.sent_tok.tokenize( textA.lower(), remove_stopwords = True ).split()  )
        textB_words = set(self.sent_tok.tokenize( textB.lower(), remove_stopwords = True ).split()  )
        
        AB_words = textA_words.intersection( textB_words )
        return float(len( AB_words ) / (  len(textA_words) + len( textB_words )  -  len( AB_words ) + 1e-12  ))


class SentenceTokenizer:
    def __init__(self ):
        pass

    def tokenize(self, sen ):
        return sen.lower()

class Vocab:
    def __init__(self, words, eos_token = "<eos>", pad_token = "<pad>", unk_token = "<unk>" ):
        self.words = words
        self.index_to_word = {}
        self.word_to_index = {}
        for idx in range( len(words) ):
            self.index_to_word[ idx ] = words[idx]
            self.word_to_index[ words[idx] ] = idx
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_index = self.word_to_index[self.eos_token]
        self.pad_index = self.word_to_index[self.pad_token]

        self.tokenizer = SentenceTokenizer()   

    def index2word( self, idx ):
        return self.index_to_word.get( idx, self.unk_token)
    def word2index( self, word ):
        return self.word_to_index.get( word, -1 )
    # The sentence needs to be tokenized 
    def sent2seq( self, sent, max_len = None , tokenize = True):
        if tokenize:
            sent = self.tokenizer.tokenize(sent)
        seq = []
        for w in sent.split():
            if w in self.word_to_index:
                seq.append( self.word2index(w) )
        if max_len is not None:
            if len(seq) >= max_len:
                seq = seq[:max_len]
            else:
                seq += [ self.pad_index ] * ( max_len - len(seq) )
        return seq
    def seq2sent( self, seq ):
        sent = []
        for i in seq:
            if i == self.eos_index or i == self.pad_index:
                break
            sent.append( self.index2word(i) )
        return " ".join(sent)

class PrefetchDataset( Dataset ):
    def __init__(self, corpus= [], 
                       paper_database = {},
                       context_database ={},
                       available_paper_ids = None,
                       max_seq_len = 512, 
                       max_doc_len = 3, 
                       words = None,
                       document_title_label = 0, 
                       document_abstract_label = 1,
                       document_fullbody_label = 2, 
                       citation_context_label = 3,
                       citation_title_label = 0, 
                       citation_abstract_label = 1,
                       citation_fullbody_label = 2,
                       padding_paragraph_label = 10,

                       max_num_samples_per_batch = 500,
                       max_n_positive = 1,
                       max_n_hard_negative = 10,
                       max_n_easy_negative = 10,


                ):
        self.corpus = corpus
        self.paper_database = paper_database
        self.context_database = context_database

        if available_paper_ids is not None:
            self.available_paper_ids = available_paper_ids
        else:
            self.available_paper_ids = list(paper_database.keys())
            
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.vocab = Vocab(words)
        self.document_title_label = document_title_label
        self.document_abstract_label = document_abstract_label
        self.document_fullbody_label = document_fullbody_label
        self.citation_context_label = citation_context_label
        self.citation_title_label = citation_title_label
        self.citation_abstract_label = citation_abstract_label
        self.citation_fullbody_label = citation_fullbody_label
        self.padding_paragraph_label = padding_paragraph_label

        self.max_num_samples_per_batch = max_num_samples_per_batch
        self.max_n_positive = max_n_positive
        self.max_n_hard_negative = max_n_hard_negative
        self.max_n_easy_negative = max_n_easy_negative

        self.jaccard_sim = JaccardSim() 


        
    def load_document( self, paper_id, is_citing_document = False ):
        paper = self.paper_database.get( paper_id, {} )
        title = paper.get("title","")
        abstract = paper.get("abstract","")
        if is_citing_document:
            document = [ [ title, self.citation_title_label ], [ abstract, self.citation_abstract_label ] ]
        else:
            document = [ [ title, self.document_title_label ], [ abstract, self.document_abstract_label ] ]
        return document

    def load_citation_context( self, context_id  ):
        context = self.context_database[ context_id ]
        context_text = context["masked_text"]
        citing_id = context["citing_id"]
        citing_document = self.load_document(citing_id, is_citing_document = True)
        citation_context_document = citing_document + [ [ context_text, self.citation_context_label ] ]
        return citation_context_document    
    
    def encode_document( self, document ):
        
        document = document[ :self.max_doc_len ]
        document = document + [ ["", self.padding_paragraph_label ] for _ in range(self.max_doc_len-len(document) ) ] 
        
        for para in document:
            if para[0].strip() == "":
                para.append(1)
            else:
                para.append(0)
        
        paragraph_text_list, paragraph_type_list, paragraph_mask_list = list(zip( *document ))
        paragraph_seq_list = [self.vocab.sent2seq(  para, self.max_seq_len ) for para in paragraph_text_list ]
        return paragraph_seq_list, paragraph_type_list, paragraph_mask_list
        
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, _ ):
        
        document_info = []
        class_label_list = []
        irrelevance_level_list = []

        while True:
            idx = np.random.choice( len(self.corpus) )
            corpus_item = self.corpus[idx]

            context_id = corpus_item["context_id"]
            citing_id = self.context_database[context_id]["citing_id"]
            document_info.append( self.encode_document( self.load_citation_context( context_id ) ) )
            query_text =  " ".join([ _[0] for _ in self.load_citation_context( context_id )])


            class_label_list.append( idx )
            irrelevance_level_list.append( 0 )

            positive_ids = corpus_item["positive_ids"]
            hard_negative_ids = corpus_item.get("prefetched_ids",[])
            hard_negative_ids = list( set( hard_negative_ids ) - set( [citing_id] + positive_ids ) )

            if "jaccard_sim_of_positive_ids" not in corpus_item:
                ## get the Jaccard similarity between the query and the positive documents
                jaccard_sim_of_positive_ids = []
                for pos_id in positive_ids:
                    document_text = " ".join( [ _[0] for _ in self.load_document( pos_id )])
                    jaccard_sim_of_positive_ids.append( self.jaccard_sim.compute_sim( query_text, document_text ) )
                jaccard_sim_of_positive_ids = np.sort( jaccard_sim_of_positive_ids )
                corpus_item["jaccard_sim_of_positive_ids"] = jaccard_sim_of_positive_ids
            else:
                jaccard_sim_of_positive_ids = corpus_item["jaccard_sim_of_positive_ids"]
                
            avg_thres_jaccard_sim = np.mean( jaccard_sim_of_positive_ids  )
            
            for pos in np.random.choice( len(positive_ids), min(len(positive_ids), self.max_n_positive ), replace = False  ):
                document_info.append( self.encode_document( self.load_document( positive_ids[pos] ) ) )
                class_label_list.append( idx )
                irrelevance_level_list.append( 1 )

            hard_negative_count = 0
            for pos in np.random.choice( len(hard_negative_ids), len(hard_negative_ids), replace = False ):
                document_text = self.load_document( hard_negative_ids[pos] )
                hard_negative_jaccard_sim = self.jaccard_sim.compute_sim( query_text,  " ".join( [_[0] for _ in document_text ] )  )
                
                if hard_negative_jaccard_sim >=  avg_thres_jaccard_sim:  
                    document_info.append( self.encode_document( document_text ) )
                    class_label_list.append( idx )
                    irrelevance_level_list.append( 2 )
                    hard_negative_count +=1
                    if hard_negative_count >= self.max_n_hard_negative:
                        break
                else: 
                    document_info.append( self.encode_document( document_text ) )
                    class_label_list.append( idx )
                    irrelevance_level_list.append( 3 )
                    hard_negative_count +=1
                    if hard_negative_count >= self.max_n_hard_negative:
                        break


            for pos in np.random.choice( len(self.available_paper_ids), self.max_n_easy_negative ):
                document_info.append( self.encode_document( self.load_document( self.available_paper_ids[pos] ) ) )
                class_label_list.append( idx )
                irrelevance_level_list.append( 3 )


            if len(irrelevance_level_list) >= self.max_num_samples_per_batch:
                break

        paragraph_seq_list, paragraph_type_list, paragraph_mask_list = list(zip(*document_info))
        paragraph_seq_list = np.asarray(paragraph_seq_list)
        paragraph_type_list = np.asarray( paragraph_type_list )
        paragraph_mask_list  = np.asarray(paragraph_mask_list) == 1
        irrelevance_level_list = np.array(irrelevance_level_list).astype(np.int32)
        class_label_list = np.array( class_label_list ).astype(np.int32)
        
        return paragraph_seq_list, paragraph_type_list, paragraph_mask_list, class_label_list, irrelevance_level_list


class PrefetchLoader:
    def __init__(self,  dset, batch_size, shuffle , worker_init_fn  , num_workers = 0, drop_last = True, pin_memory = True):
        self.dataloader = DataLoader( dset, batch_size= batch_size, shuffle= shuffle, 
                                     worker_init_fn = worker_init_fn , 
                                     num_workers= num_workers , drop_last= drop_last, pin_memory = pin_memory)
        def cycle(dataloader):
            while True:
                for x in dataloader:
                    yield x
        self.dataiter = iter(cycle( self.dataloader ))
    def get_next( self ):
        return next(self.dataiter ) 
