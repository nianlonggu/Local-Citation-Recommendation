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
    parser.add_argument("-signal_file", default = None)
    parser.add_argument("-shuffle_seed", type = int, default = None   )

    args_input = parser.parse_args()
    args = Dict2Class(json.load(open(args_input.config_file_path)))

    signal_file = args_input.signal_file

    if args_input.start is not None:
        args.start = args_input.start
    if args_input.size is not None:
        args.size = args_input.size 
    if args_input.encoder_gpu_list is not None:
        args.encoder_gpu_list = args_input.encoder_gpu_list
    if args_input.shuffle_seed is not None:
        args.shuffle_seed = args_input.shuffle_seed

    if not( args.start == 0 and args.size == 0):
        args.prefetch_embedding_path = args.prefetch_embedding_path +"_%d_%d"%( args.start, args.size )
    
    with open(args.unigram_words_path,"rb") as f:
        words = pickle.load(f)
    prefetch_dataset = PrefetchDataset(words = words)

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

    # assert ckpt_name is not None

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
        
    ## compute embedding
    paper_database = json.load(open(args.paper_database_path))

    if args.available_paper_ids_path is not None:
        available_paper_ids = json.load(open(args.available_paper_ids_path))
    else:
        #available_paper_ids = list( paper_database.keys() )
        available_paper_ids = []
        for pid in paper_database:
            if paper_database.get( "is_in_paper_database", True ):
                available_paper_ids.append(pid)

    available_paper_ids.sort()

    if args.shuffle_seed is not None:
        np.random.seed( args.shuffle_seed )
        np.random.shuffle( available_paper_ids )

    embeddings = []
    index_to_id_mapper = {}
    id_to_index_mapper = {}
    
    paragraphs_cache = []
    
    for index, paperid in enumerate( tqdm( available_paper_ids ) ):
        if index < args.start:
            continue
        if args.size >0 and index >= args.start + args.size:
            break

        paper = paper_database[paperid]
        paragraphs = [ [ paper["title"], prefetch_dataset.document_title_label ], [paper["abstract"],prefetch_dataset.document_abstract_label ]  ]        
        paragraphs_cache.append(paragraphs)
        if len(paragraphs_cache) >= args.paragraphs_cache_size:
            embeddings.append(encoder.encode( paragraphs_cache ))
            paragraphs_cache = []

        index_to_id_mapper[index] = paperid
        id_to_index_mapper[paperid] = index

    if len(paragraphs_cache) > 0:
        embeddings.append(encoder.encode( paragraphs_cache ))
        paragraphs_cache = []
    
    if len(embeddings)>0:
        embeddings = np.concatenate( embeddings, axis = 0 )
    else:
        embeddings = np.zeros(  (0, args.embed_dim ) ).astype(np.float32)

    with open(ensure_dir_exists(embedding_path),"wb") as f:
        pickle.dump( {
            "index_to_id_mapper":index_to_id_mapper,
            "id_to_index_mapper":id_to_index_mapper,
            "embedding":embeddings
        }, f, -1 )

    if signal_file is not None:
        with open(signal_file,"a") as f:
            f.write("Done!\n")
