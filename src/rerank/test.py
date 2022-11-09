from utils import *
from datautils import *
from model import *
import os
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import torch.nn.functional as F
import time
import pickle
import numpy as np
from transformers import AutoTokenizer

import argparse

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
        args.rerank_results_save_path = args.rerank_results_save_path+"_%d_%d"%( args.start, args.size )
    try:
        if not os.path.exists(args.model_folder):
            os.makedirs(args.model_folder)
        if not os.path.exists(args.log_folder):
            os.makedirs(args.log_folder)
    except:
        pass
    print("restoring checkpoint ...", flush = True)

    # restore most recent checkpoint
    if args.model_path is None:
        ckpt, ckpt_name = load_model( args.model_folder,  return_ckpt_name = True )
    else:
        ckpt = torch.load( args.model_path,  map_location=torch.device('cpu') )
        ckpt_name = args.model_path
    assert ckpt is not None

    LOG( ckpt_name )

    tokenizer = AutoTokenizer.from_pretrained( args.initial_model_path )
    tokenizer.add_special_tokens( { 'additional_special_tokens': ['<cit>','<sep>','<eos>'] } )
                        
    corpus = json.load( open(args.corpus_path, "r") )

    if args.size == 0:
        corpus = corpus[args.start:]
    else:
        corpus = corpus[args.start:args.start + args.size]
    paper_database = json.load(open(args.paper_database_path))

    context_database = json.load( open(args.context_database_path) )

    rerank_dataset = RerankDataset( corpus, paper_database, context_database, tokenizer,
                                   rerank_top_K = args.rerank_top_K,
                                   max_input_length = args.max_input_length,
                                   is_training = False
                                    )
    rerank_dataloader = DataLoader( rerank_dataset, batch_size= args.n_query_per_batch, shuffle= False, 
                                  num_workers= args.num_workers,  drop_last= False )

    vocab_size = len(tokenizer)
    scorer = Scorer( args.initial_model_path, vocab_size )
    scorer.load_state_dict( ckpt["scorer"] )
    
    if args.gpu_list is not None:
        assert len(args.gpu_list) == args.n_device
    else:
        args.gpu_list = np.arange(args.n_device).tolist()
    device = torch.device( "cuda:%d"%(args.gpu_list[0]) if torch.cuda.is_available() else "cpu"  )
    scorer.to(device)

    if device.type == "cuda" and args.n_device > 1:
        scorer = nn.DataParallel( scorer, args.gpu_list )

    sorted_irrelevance_levels_list = []
    num_positive_ids_list = []

    query_time_list = []

    print("starting test ...", flush = True)

    for count, batch in enumerate(tqdm(rerank_dataloader)):

        irrelevance_levels = batch["irrelevance_levels"].to(device)
        input_ids =  batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        num_positive_ids = batch[ "num_positive_ids" ] 
        n_doc = input_ids.size(1)

        num_positive_ids_list += num_positive_ids.detach().cpu().numpy().tolist()

        input_ids = input_ids.view(-1,input_ids.size(2))
        token_type_ids = token_type_ids.view(-1,token_type_ids.size(2))
        attention_mask = attention_mask.view(-1, attention_mask.size(2))

        tic = time.time()
        score = []
        for pos in range( 0, input_ids.size(0), args.sub_batch_size ):
            with torch.no_grad():
                score.append( scorer( 
                    {
                        "input_ids":input_ids[pos:pos+args.sub_batch_size] ,
                        "token_type_ids":token_type_ids[pos:pos+args.sub_batch_size] ,
                        "attention_mask":attention_mask[pos:pos+args.sub_batch_size]
                    }  ).detach() )
        score = torch.cat( score, dim =0 ).view( -1, n_doc ).cpu().numpy()

        tac = time.time()
        query_time_list.append( tac - tic )


        irrelevance_levels = irrelevance_levels.detach().cpu().numpy()

        for pos in range( irrelevance_levels.shape[0] ):
            sorted_irrelevance_level = list( zip( *sorted( zip( irrelevance_levels[pos], score[pos] ), key = lambda x:-x[1] ) ) )[0]
            sorted_irrelevance_levels_list.append(sorted_irrelevance_level)


    if not args_input.time_analysis:

        num_positive_ids_arr = np.array( num_positive_ids_list )
        assert np.all( num_positive_ids_arr >0 )
        ## make sure no irrelevance_level is -1
        max_irrelevance_levels_len = np.max( [ len( item ) for item in sorted_irrelevance_levels_list  ]  )
        sorted_irrelevance_levels_arr = [ list( item) + [-1] * ( max_irrelevance_levels_len-len(item) )    for item in  sorted_irrelevance_levels_list ]
        sorted_irrelevance_levels_arr = np.asarray( sorted_irrelevance_levels_arr )
        hit_matrix = sorted_irrelevance_levels_arr == rerank_dataset.irrelevance_level_for_positive
        precision_at_K = {}
        recall_at_K ={}
        F_at_K = {}
        for K in args.K_list:
            recall_at_K[K] = (hit_matrix[:,:K].sum(axis = 1)/num_positive_ids_arr   ).mean()
            precision_at_K[K] = (hit_matrix[:,:K].mean(axis = 1) ).mean()
            F_at_K[K] = 2/( 1/(recall_at_K[K] + 1e-12) + 1/(precision_at_K[K]  + 1e-12  ) )
        print({ "precision":precision_at_K, "recall":recall_at_K, "F1": F_at_K })
        LOG( json.dumps( { "precision":precision_at_K, "recall":recall_at_K, "F1": F_at_K } ) )
        with open( ensure_dir_exists( args.rerank_results_save_path ), "wb" ) as f:
            pickle.dump( [ hit_matrix, num_positive_ids_arr  ], f, -1 )
        print("Finished!")
        LOG("Finished!")
    else:

        print( "Time Analysis Results:" )
        print( "%.1f ± %.1f ms" % ( np.mean( query_time_list ) * 1000, np.std( query_time_list ) * 1000  )  )











