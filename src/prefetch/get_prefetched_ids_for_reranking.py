from rankers import *
from utils import *
from datautils import PrefetchDataset
import os
import pickle
from glob import glob
import numpy as np
import argparse
import json


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file_path" )
    parser.add_argument("-start", type = int, default = None)
    parser.add_argument("-size", type = int, default = None)
    parser.add_argument("-encoder_gpu_list", type = int, nargs = "+", default = None)
    parser.add_argument("-ranker_gpu_list", type = int, nargs = "+", default = None)
    parser.add_argument("-signal_file", default = None)
    parser.add_argument("-shuffle_seed", type = int, default = None   )
    parser.add_argument("-output_corpus_path", default = None)

    args_input = parser.parse_args()
    args = Dict2Class(json.load(open(args_input.config_file_path)))

    signal_file = args_input.signal_file

    if args_input.start is not None:
        args.start = args_input.start
    if args_input.size is not None:
        args.size = args_input.size 
    if args_input.encoder_gpu_list is not None:
        args.encoder_gpu_list = args_input.encoder_gpu_list
    if args_input.ranker_gpu_list is not None:
        args.ranker_gpu_list = args_input.ranker_gpu_list
    if args_input.shuffle_seed is not None:
        args.shuffle_seed = args_input.shuffle_seed
    if args_input.output_corpus_path is not None:
        args.output_corpus_path = args_input.output_corpus_path

    # if not( args.start == 0 and args.size == 0):
    args.output_corpus_path = args.output_corpus_path +"_%d_%d"%( args.start, args.size )

    with open(args.unigram_words_path,"rb") as f:
        words = pickle.load(f)
    prefetch_dataset = PrefetchDataset(words = words)
    

    if args.prefetch_model_path is not None:
        ckpt_name = args.prefetch_model_path
    else:
        try:
            if not os.path.exists( args.prefetch_model_folder ):
                ckpt_name = None
            else:
                ckpt_list =  glob( args.prefetch_model_folder + "/*.pt" )
                if len( ckpt_list ) >0:
                    ckpt_list.sort( key = os.path.getmtime )
                    ckpt_name = ckpt_list[-1]
                else:
                    ckpt_name = None
        except:
            ckpt_name = None
    
    if ckpt_name is None:
        use_BM25 = True
        with open(args.inverted_index_path, "rb") as f:
            inv_idx_info = pickle.load(f)
        ranker = BM25Ranker(inv_idx_info)

    else:
        use_BM25 = False
        encoder = PrefetchEncoder( ckpt_name, args.unigram_words_path, 
                                   args.embed_dim, args.encoder_gpu_list,
                                   args.num_heads, args.hidden_dim,  
                                   args.max_seq_len, args.max_doc_len, 
                                   args.n_para_types, args.num_enc_layers
                                 )
        embedding_path = args.prefetch_embedding_path
        
        ranker = Ranker( embedding_path, args.embed_dim , gpu_list = args.ranker_gpu_list )
        ranker.encoder = encoder


    paper_database = json.load(open(args.paper_database_path))

    if args.cr_mode == "local":
        context_database = json.load( open( args.context_database_path ) )
    
    corpus = json.load(open(args.input_corpus_path))

    if args.shuffle_seed is not None:
        np.random.seed( args.shuffle_seed )
        np.random.shuffle( corpus )

    if args.size == 0:
        corpus = corpus[args.start:]
    else:
        corpus = corpus[args.start:args.start + args.size]

    top_K = args.top_K
    recall_list = []
    for example in tqdm(corpus):

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

        if use_BM25:
            query_text = " ".join( [ item[0]  for item in query_text[0] ] )


        candidates = ranker.get_top_n( top_K+1, query_text )
        
        if citing_id in set(candidates):
            candidates.remove( citing_id )
        
        example["prefetched_ids"] = candidates[:top_K]


    with open(  args.output_corpus_path,"w" ) as f:
        for example in corpus:
            f.write( json.dumps( example ) + "\n" )


    if signal_file is not None:
        with open(signal_file,"a") as f:
            f.write("Done!\n")
    
    

    
    
    

