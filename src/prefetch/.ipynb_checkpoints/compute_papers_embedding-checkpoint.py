from rankers import *
from utils import *
import os
import pickle
from glob import glob
import numpy as np
import argparse
import json


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-unigram_words_path")
    parser.add_argument("-prefetch_model_folder")
    parser.add_argument("-prefetch_model_path", default = None)
    parser.add_argument("-prefetch_embedding_path")
    parser.add_argument("-paper_database_path")
    parser.add_argument("-paragraphs_cache_size", type = int, default = 16)
    parser.add_argument("-shuffle", type = int, default = 0)
    parser.add_argument("-start", type = int, default = 0)
    parser.add_argument("-size", type = int, default = 0)
    parser.add_argument("-encoder_gpu_list", type = int, nargs = "+", default = None)
    parser.add_argument("-embed_dim", type = int, default = 200)
    parser.add_argument("-num_heads", type = int, default = 8)
    parser.add_argument("-hidden_dim", type = int, default = 1024)
    parser.add_argument("-max_seq_len", type = int, default = 512)
    parser.add_argument("-max_doc_len", type = int, default = 3)
    parser.add_argument("-n_para_types", type = int, default = 100)
    parser.add_argument("-num_enc_layers", type = int, default = 1)
    parser.add_argument("-document_title_label", type = int, default = 0)
    parser.add_argument("-document_abstract_label", type = int, default = 1)
    parser.add_argument("-document_fullbody_label", type = int, default = 2)
    

    args = parser.parse_args()
    
    paper_database = json.load(open(args.paper_database_path))

    available_paper_ids = list( paper_database.keys() )
    available_paper_ids.sort()
    if args.shuffle:
        np.random.shuffle( available_paper_ids )
    
    if args.size == 0:
        args.size = len(available_paper_ids) 

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

        
    ## compute embedding
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
        paragraphs = [ [ paper["title"], args.document_title_label ], [paper["abstract"],args.document_abstract_label ]  ]        
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

    with open(ensure_dir_exists(args.prefetch_embedding_path),"wb") as f:
        pickle.dump( {
            "index_to_id_mapper":index_to_id_mapper,
            "id_to_index_mapper":id_to_index_mapper,
            "embedding":embeddings
        }, f, -1 )
