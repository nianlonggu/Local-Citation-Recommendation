from utils import *
from losses import *
from datautils import *
from model import *
from transformers import AdamW, AutoTokenizer
import os
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import torch.nn.functional as F
import time
import numpy as np
import argparse

def LOG( info, end="\n" ):
    with open( args.log_folder + "/"+ args.log_file_name , "a" ) as f:
        f.write( info + end )


def train_iteration(batch):

    irrelevance_levels = batch["irrelevance_levels"].to(device)
    input_ids =  batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    n_doc = input_ids.size(1)
    score = scorer( 
        {
            "input_ids":input_ids.view( -1, input_ids.size(2) ),
            "token_type_ids":token_type_ids.view( -1, token_type_ids.size(2) ),
            "attention_mask":attention_mask.view( -1, attention_mask.size(2) )
        } )
    score = score.view(-1, n_doc)
    loss = triplet_loss(score, irrelevance_levels )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    return loss.item()

def validate_iteration(batch):

    irrelevance_levels = batch["irrelevance_levels"].to(device)
    input_ids =  batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)


    n_doc = input_ids.size(1)
    with torch.no_grad():
        score = scorer( {
            "input_ids":input_ids.view( -1, input_ids.size(2) ),
            "token_type_ids":token_type_ids.view( -1, token_type_ids.size(2) ),
            "attention_mask":attention_mask.view( -1, attention_mask.size(2) )
        } )
        score = score.view(-1, n_doc)
        loss = triplet_loss(score, irrelevance_levels )
    return loss.item()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file_path" )
    args_input = parser.parse_args()

    args = Dict2Class(json.load(open(args_input.config_file_path)))


    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)

    # restore most recent checkpoint
    if args.restore_old_checkpoint:
        ckpt = load_model( args.model_folder )
    else:
        ckpt = None

    tokenizer = AutoTokenizer.from_pretrained( args.initial_model_path )
    tokenizer.add_special_tokens( { 'additional_special_tokens': ['<cit>','<sep>','<eos>'] } )
    
    corpus = json.load(open( args.train_corpus_path, "r" ))
                

    paper_database = json.load(open(args.paper_database_path))

    context_database = json.load( open(args.context_database_path) )

        
    rerank_dataset = RerankDataset( corpus, paper_database, context_database, tokenizer,
                                  rerank_top_K = args.rerank_top_K,
                                  max_input_length = args.max_input_length,
                                  is_training = True,
                                  n_document= args.n_document, 
                                  max_n_positive = args.max_n_positive,
                                  )
    rerank_dataloader = DataLoader( rerank_dataset, batch_size= args.n_query_per_batch, shuffle= True, 
                                  num_workers= args.num_workers,  drop_last= True, 
                                  worker_init_fn = lambda x:[np.random.seed( int( time.time() )+x ), torch.manual_seed(int( time.time() ) + x) ],
                                  pin_memory= True )

    
    val_corpus = json.load( open(args.val_corpus_path, "r") )

    val_rerank_dataset = RerankDataset( val_corpus, paper_database, context_database, tokenizer,
                                  rerank_top_K = args.rerank_top_K,
                                  max_input_length = args.max_input_length,
                                  is_training = True,
                                  n_document= args.n_document, 
                                  max_n_positive = args.max_n_positive,
                                  )
    val_rerank_dataloader = DataLoader( val_rerank_dataset, batch_size= args.n_query_per_batch, shuffle= False, 
                                  num_workers= args.num_workers,  drop_last= True, 
                                  worker_init_fn = lambda x:[np.random.seed( int( time.time() )+x ), torch.manual_seed(int( time.time() ) + x) ],
                                  pin_memory= True )



    vocab_size = len(tokenizer)
    scorer = Scorer( args.initial_model_path, vocab_size )

    if ckpt is not None:
        scorer.load_state_dict( ckpt["scorer"] )
        LOG("model restored!")
        print("model restored!")
    
    if args.gpu_list is not None:
        assert len(args.gpu_list) == args.n_device
    else:
        args.gpu_list = np.arange(args.n_device).tolist()

    device = torch.device( "cuda:%d"%(args.gpu_list[0]) if torch.cuda.is_available() else "cpu"  )
    scorer.to(device)

    if device.type == "cuda" and args.n_device > 1:
        scorer = nn.DataParallel( scorer, args.gpu_list )
        model_parameters = [ par for par in scorer.module.parameters() if par.requires_grad  ] 
    else:
        model_parameters = [ par for par in scorer.parameters() if par.requires_grad  ] 
    optimizer = AdamW( model_parameters , lr= args.initial_learning_rate,  weight_decay = args.l2_weight  ) 

    if ckpt is not None:
        optimizer.load_state_dict( ckpt["optimizer"] )
        LOG("optimizer restored!")
        print("optimizer restored!")

    current_batch = 0
    if ckpt is not None:
        current_batch = ckpt["current_batch"]
        LOG("current_batch restored!")
        print("current_batch restored!")
    running_losses = []

    triplet_loss = TripletLoss(args.base_margin)
    for epoch in range(args.num_epochs):
        for count, batch in enumerate(tqdm(rerank_dataloader)):
            current_batch +=1

            loss = train_iteration( batch )

            running_losses.append(loss)

            if current_batch % args.print_every == 0:
                print("[batch: %05d] loss: %.4f"%( current_batch, np.mean(running_losses) ))
                LOG( "[batch: %05d] loss: %.4f"%( current_batch, np.mean(running_losses) ) )
                os.system( "nvidia-smi > %s/gpu_usage.log"%( args.log_folder ) )
                running_losses = []
            if current_batch % args.save_every == 0 :  
                save_model(  { 
                    "current_batch":current_batch,
                    "scorer": scorer,
                    "optimizer": optimizer.state_dict()
                    } ,  args.model_folder+"/model_batch_%d.pt"%( current_batch ), 10 )
                print("Model saved!")
                LOG("Model saved!")

            if current_batch % args.validate_every == 0:
                running_losses_val = []
                for val_count, batch in enumerate(tqdm(val_rerank_dataloader)):
                    loss = validate_iteration( batch )
                    running_losses_val.append(loss)

                    if val_count >= args.num_validation_iterations:
                        break
                print("[batch: %05d] validation loss: %.4f"%( current_batch, np.mean( running_losses_val )  ) )
                LOG("[batch: %05d] validation loss: %.4f"%( current_batch, np.mean( running_losses_val )  ) )
                

        running_losses_val = []
        for val_count, batch in enumerate(tqdm(val_rerank_dataloader)):
            loss = validate_iteration( batch )
            running_losses_val.append(loss)
            if val_count >= args.num_validation_iterations:
                break
        print("[batch: %05d] validation loss: %.4f"%( current_batch, np.mean( running_losses_val )  ) )
        LOG("[batch: %05d] validation loss: %.4f"%( current_batch, np.mean( running_losses_val )  ) )
                


        save_model(  { 
                    "current_batch":current_batch,
                    "scorer": scorer,
                    "optimizer": optimizer.state_dict()
                    } ,  args.model_folder+"/model_batch_%d.pt"%( current_batch ), args.max_num_checkpoints )
        print("Model saved!")
        LOG("Model saved!")
