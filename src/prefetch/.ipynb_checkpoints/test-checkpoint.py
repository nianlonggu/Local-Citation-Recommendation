from rankers import *
from utils import *
import os
import pickle
from glob import glob
import numpy as np
import argparse
import json
import time

def LOG( info, end="\n" ):
    with open( args.log_folder + "/"+ args.log_file_name , "a" ) as f:
        f.write( info + end )


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-log_folder")
    parser.add_argument("-log_file_name")
    parser.add_argument("-start", type = int, default = 0)
    parser.add_argument("-size", type = int, default = 0)
    parser.add_argument("-K_list", type = int, nargs = "+", default = [10,20,50,100,200,500,1000,2000] )
    parser.add_argument("-encoder_gpu_list", type = int, nargs = "+", default = None)
    parser.add_argument("-ranker_gpu_list", type = int, nargs = "+", default = None)
    parser.add_argument("-input_corpus_path")
    parser.add_argument("-unigram_words_path")
    parser.add_argument("-prefetch_model_folder")
    parser.add_argument("-prefetch_model_path", default = None)
    parser.add_argument("-prefetch_embedding_path")
    parser.add_argument("-paper_database_path")
    parser.add_argument("-context_database_path")
    parser.add_argument("-embed_dim", type = int, default = 200)
    parser.add_argument("-num_heads", type = int, default = 8)
    parser.add_argument("-hidden_dim", type = int, default = 1024)
    parser.add_argument("-max_seq_len", type = int, default = 512)
    parser.add_argument("-max_doc_len", type = int, default = 3)
    parser.add_argument("-n_para_types", type = int, default = 100)
    parser.add_argument("-num_enc_layers", type = int, default = 1)
    parser.add_argument("-citation_title_label", type = int, default = 0)
    parser.add_argument("-citation_abstract_label", type = int, default = 1)
    parser.add_argument("-citation_context_label", type = int, default = 3)
    
    args = parser.parse_args()

    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    

    if args.prefetch_model_path is not None:
        ckpt_name = args.prefetch_model_path
    else:
        try:
            ckpt_list =  glob( args.prefetch_model_folder + "/*.pt" )
            if len( ckpt_list ) >0:
                ckpt_list.sort( key = os.path.getmtime )
                ckpt_name = ckpt_list[-1]
            else:
                ckpt_name = None
        except:
            ckpt_name = None
        
    assert ckpt_name is not None
    encoder = PrefetchEncoder( ckpt_name, args.unigram_words_path, 
                                       args.embed_dim, args.encoder_gpu_list,
                                       args.num_heads, args.hidden_dim,  
                                       args.max_seq_len, args.max_doc_len, 
                                       args.n_para_types, args.num_enc_layers
                                     )
            
    ranker = Ranker( args.prefetch_embedding_path, args.embed_dim , gpu_list = args.ranker_gpu_list )
    ranker.encoder = encoder

    paper_database = json.load(open(args.paper_database_path))
    corpus = json.load(open(args.input_corpus_path))
    if args.size == 0:
        corpus = corpus[args.start:]
    else:
        corpus = corpus[args.start:args.start + args.size]

    context_database = json.load(open( args.context_database_path ))
        
    ###################################
    K_list = args.K_list
    max_K = np.max(K_list)

    positive_ids_list = []
    candidates_list = []
    query_time_list = []
    
    for count, example in enumerate(tqdm(corpus)):

        context_id = example["context_id"]
        citing_id = context_database[context_id]["citing_id"]
        citing_paper_info  = paper_database.get(citing_id,{} )
        query_text = [
                        [ 
                            [ citing_paper_info.get("title",""), args.citation_title_label ],
                            [ citing_paper_info.get("abstract",""), args.citation_abstract_label ],
                     [ context_database[ context_id ]["masked_text"], args.citation_context_label ]  
                        ]  
                    ]

        tic = time.time()
        candidates = ranker.get_top_n( max_K+1, query_text )
        tac = time.time()
        
        query_time_list.append( tac - tic )
        
        if citing_id in candidates and citing_id not in set( example["positive_ids"] ) :
            candidates.remove( citing_id)
        candidates = candidates[:max_K]

        positive_ids_list.append(example["positive_ids"])
        candidates_list.append(candidates)
            
    precision_at_K = {}
    recall_at_K ={}
    F_at_K = {}
    for K in tqdm(K_list):
        recall_list = []
        precision_list = []
        for positive_ids, candidates in zip(positive_ids_list, candidates_list):
            hit_num = len(set(positive_ids) & set( candidates[:K] ))
            recall_list.append( hit_num / len( set(positive_ids) ) )
            precision_list.append( hit_num / K )
        recall_at_K[K] = np.mean(recall_list)
        precision_at_K[K] = np.mean( precision_list )
        F_at_K[K] = 2/( 1/(recall_at_K[K] + 1e-12) + 1/(precision_at_K[K]  + 1e-12  ) )
    ckpt_name = str(ckpt_name)
    print({ "ckpt_name": ckpt_name, "recall":recall_at_K }, flush = True)
    LOG( json.dumps( { "ckpt_name": ckpt_name, "recall":recall_at_K } ) )
    print("Finished!", flush = True)
    LOG("Finished!")


