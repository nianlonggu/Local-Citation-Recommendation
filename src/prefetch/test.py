from rankers import *
from utils import *
from datautils import PrefetchDataset
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
    parser.add_argument("-config_file_path" )
    parser.add_argument("-start", type = int, default = None)
    parser.add_argument("-size", type = int, default = None)
    parser.add_argument("-time_analysis", type = int, default = 0)
    args_input = parser.parse_args()
    args = Dict2Class(json.load(open(args_input.config_file_path)))

    if args_input.start is not None:
        args.start = args_input.start
    if args_input.size is not None:
        args.size = args_input.size 

    if not( args.start == 0 and args.size == 0):
        args.prefetch_results_save_path = args.prefetch_results_save_path+"_%d_%d"%( args.start, args.size )

    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    
    with open(args.unigram_words_path,"rb") as f:
        words = pickle.load(f)
    prefetch_dataset = PrefetchDataset(words = words)


    if args.use_BM25:
        with open(args.inverted_index_path, "rb") as f:
            inv_idx_info = pickle.load(f)
        ranker = BM25Ranker(inv_idx_info)
        ckpt_name = None
    else:
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
        
        if ckpt_name is not None:
            encoder = PrefetchEncoder( ckpt_name, args.unigram_words_path, 
                                       args.embed_dim, args.encoder_gpu_list,
                                       args.num_heads, args.hidden_dim,  
                                       args.max_seq_len, args.max_doc_len, 
                                       args.n_para_types, args.num_enc_layers
                                     )
        else:
            encoder = Sent2vecEncoder( args.sent2vec_model_path )
            
        embedding_path = args.prefetch_embedding_path
            
        ranker = Ranker( embedding_path, args.embed_dim , gpu_list = args.ranker_gpu_list )
        ranker.encoder = encoder

    paper_database = json.load(open(args.paper_database_path))
    corpus = json.load(open(args.corpus_path))
    if args.size == 0:
        corpus = corpus[args.start:]
    else:
        corpus = corpus[args.start:args.start + args.size]

    if args.cr_mode == "local":
        context_database = json.load(open( args.context_database_path ))
        
    ###################################
    K_list = args.K_list
    max_K = np.max(K_list)

    positive_ids_list = []
    candidates_list = []
    
    query_time_list = []
    
    for count, example in enumerate(tqdm(corpus)):

        if args.cr_mode == "local":
            context_id = example["context_id"]
            citing_id = context_database[context_id]["citing_id"]
            citing_paper_info  = paper_database.get(citing_id,{} )
            query_text = [
                                [ 
                                  [ citing_paper_info.get("title",""), prefetch_dataset.citation_title_label ],
                                  [ citing_paper_info.get("abstract",""), prefetch_dataset.citation_abstract_label ],
                                  [ context_database[ context_id ]["masked_text"], prefetch_dataset.citation_context_label ]  
                                ]  
                            ]
        elif args.cr_mode == "global":
            citing_id = example["id"]
            citing_paper_info = paper_database.get( citing_id, {} )
            query_text = [
                        [ 
                            [ citing_paper_info.get("title",""), prefetch_dataset.citation_title_label ],
                            [ citing_paper_info.get("abstract",""), prefetch_dataset.citation_abstract_label ],
                        ]  
                    ]

        if args.use_BM25:
            query_text = " ".join( [ item[0]  for item in query_text[0] ] )
        
        tic = time.time()
        candidates = ranker.get_top_n( max_K+1, query_text )
        tac = time.time()
        
        query_time_list.append( tac - tic )
        
        
        if citing_id in candidates and citing_id not in set( example["positive_ids"] ) :
            candidates.remove( citing_id)
        candidates = candidates[:max_K]

        positive_ids_list.append(example["positive_ids"])
        candidates_list.append(candidates)

        if not args_input.time_analysis and args.print_every >0 and count>0 and count % args.print_every == 0:
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

            print({ "precision":precision_at_K, "recall":recall_at_K, "F1": F_at_K })
            LOG( json.dumps( { "precision":precision_at_K, "recall":recall_at_K, "F1": F_at_K } ) )

            
    if not args_input.time_analysis:
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
        print({ "ckpt_name": ckpt_name, "precision":precision_at_K, "recall":recall_at_K, "F1": F_at_K }, flush = True)
        LOG( json.dumps( { "ckpt_name": ckpt_name, "precision":precision_at_K, "recall":recall_at_K, "F1": F_at_K } ) )
        with open( ensure_dir_exists( args.prefetch_results_save_path ), "wb" ) as f:
            pickle.dump(  [positive_ids_list, candidates_list], f, -1 )
        print("Finished!", flush = True)
        LOG("Finished!")
    else:
        print("Time Analysis Results:")
        print("%.3f ± %.3f ms"%( np.mean( query_time_list ) * 1000, np.std(  query_time_list ) * 1000  ) )
    

