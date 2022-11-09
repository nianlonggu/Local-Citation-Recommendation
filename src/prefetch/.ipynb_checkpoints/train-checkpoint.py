from model import *
from utils import *
from datautils import *
from losses import *
import json
import time
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import copy

import argparse



def update_moving_average( m_ema, m, decay ):
    with torch.no_grad():
        param_dict_m_ema =  m_ema.module.parameters()  if isinstance(  m_ema, nn.DataParallel ) else m_ema.parameters() 
        param_dict_m =  m.module.parameters()  if isinstance( m , nn.DataParallel ) else  m.parameters() 
        for param_m_ema, param_m in zip( param_dict_m_ema, param_dict_m ):
            param_m_ema.copy_( decay * param_m_ema + (1-decay) *  param_m )

def LOG( info, end="\n" ):
    with open( args.log_folder + "/"+ args.log_file_name , "a" ) as f:
        f.write( info + end )

def train_iteration( batch ):
    paragraph_seq_list, paragraph_type_list, paragraph_mask_list, class_label_list, irrelevance_level_list = [ item[0].to(device) for item in batch]
    doc_embedding = document_encoder(paragraph_seq_list,paragraph_type_list,paragraph_mask_list, args.dropout_rate  )
    loss = triplet_loss( doc_embedding, class_label_list, irrelevance_level_list, args.positive_irrelevance_levels , args.similarity )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return loss.item()

def validate_iteration( batch ):
    paragraph_seq_list, paragraph_type_list, paragraph_mask_list, class_label_list, irrelevance_level_list = [ item[0].to(device) for item in batch]
    n_doc = paragraph_seq_list.size(1)
    with torch.no_grad():
        doc_embedding = document_encoder_ema(paragraph_seq_list ,
                                         paragraph_type_list,
                                         paragraph_mask_list )
        
        loss = triplet_loss( doc_embedding, class_label_list, irrelevance_level_list,  args.positive_irrelevance_levels , args.similarity )

    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-unigram_words_path"  )
    parser.add_argument("-unigram_embedding_path"  )
    parser.add_argument("-prefetch_model_folder"  )
    parser.add_argument("-paper_database_path"  )
    parser.add_argument("-context_database_path"  )
    parser.add_argument("-train_corpus_path"  )
    parser.add_argument("-log_folder"  )
    parser.add_argument("-log_file_name"  )
    parser.add_argument("-max_num_samples_per_batch", type = int  )
    parser.add_argument("-n_device", type = int, default = 1  )
    parser.add_argument("-gpu_list", type = int, nargs = "+", default = None)
    parser.add_argument("-print_every", type = int, default = 100  )
    parser.add_argument("-save_every", type = int, default = 200  )
    parser.add_argument("-max_num_iterations", type = int, default = 200  )
    
    parser.add_argument("-max_n_positive", type = int, default = 1 )
    parser.add_argument("-max_n_hard_negative", type = int, default = 3  )
    parser.add_argument("-max_n_easy_negative", type = int, default = 1  )
    parser.add_argument("-num_workers", type = int, default = 2  )
    parser.add_argument("-initial_learning_rate", type = float, default = 1e-4 )
    parser.add_argument("-l2_weight",  type = float, default = 1e-5  )
    parser.add_argument("-dropout_rate",  type = float, default = 0.1  )
    parser.add_argument("-moving_average_decay",  type = float, default = 0.999  )
    parser.add_argument("-base_margin",  type = float, default = 0.05  )
    parser.add_argument("-similarity", default = "cosine"  )
    parser.add_argument("-positive_irrelevance_levels", type = int, nargs = "+", default = [1,2])
    parser.add_argument("-max_num_checkpoints", type = int, default = 20  )
    parser.add_argument("-embed_dim", type = int, default = 200  )
    parser.add_argument("-num_heads", type = int, default = 8 )
    parser.add_argument("-hidden_dim", type = int, default = 1024  )
    parser.add_argument("-max_seq_len", type = int, default = 512  )
    parser.add_argument("-max_doc_len", type = int, default = 3  )
    parser.add_argument("-n_para_types", type = int, default = 100  )
    parser.add_argument("-num_enc_layers", type = int, default = 1  )
    parser.add_argument("-document_title_label", type = int, default = 0  )
    parser.add_argument("-document_abstract_label", type = int, default = 1  )
    parser.add_argument("-document_fullbody_label", type = int, default = 2  )
    parser.add_argument("-citation_title_label", type = int, default = 0  )
    parser.add_argument("-citation_abstract_label", type = int, default = 1  )
    parser.add_argument("-citation_fullbody_label", type = int, default = 2  )
    parser.add_argument("-citation_context_label", type = int, default = 3  )
    parser.add_argument("-padding_paragraph_label", type = int, default = 10  )
    
    
    args = parser.parse_args()

    
    unigram_embedding = pickle.load( open( args.unigram_embedding_path, "rb" ) )
    words = pickle.load( open( args.unigram_words_path, "rb" ) )

    vocab = Vocab(words )
    embed_dim = unigram_embedding.shape[1]
    vocab_size = unigram_embedding.shape[0]
    pad_index = vocab.pad_index
    if not os.path.exists(args.prefetch_model_folder):
        os.makedirs(args.prefetch_model_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)

    paper_database = json.load(open(args.paper_database_path))
    train_corpus = json.load(open(args.train_corpus_path))


    context_database = json.load(open( args.context_database_path ))
    available_paper_ids = None
    
    train_dataset = PrefetchDataset( train_corpus, paper_database, context_database, available_paper_ids, 
                                     args.max_seq_len, 
                                     args.max_doc_len, 
                                     words, 
                                    
                                     document_title_label = args.document_title_label, 
                                     document_abstract_label = args.document_abstract_label,
                                     document_fullbody_label = args.document_fullbody_label, 
                                     citation_context_label = args.citation_context_label,
                                     citation_title_label = args.citation_title_label, 
                                     citation_abstract_label = args.citation_abstract_label,
                                     citation_fullbody_label = args.citation_fullbody_label,
                                     padding_paragraph_label = args.padding_paragraph_label,
                                    
                                     max_num_samples_per_batch = args.max_num_samples_per_batch,
                                     max_n_positive = args.max_n_positive,
                                     max_n_hard_negative = args.max_n_hard_negative,
                                     max_n_easy_negative = args.max_n_easy_negative

                                      )
    train_dataloader = PrefetchLoader( train_dataset, batch_size= 1, shuffle= True, 
                                  worker_init_fn = lambda x:[np.random.seed( int( time.time() )+x ), torch.manual_seed(int( time.time() ) + x) ],
                                  num_workers= args.num_workers,  drop_last= True, 
                                  pin_memory= True )



    # restore most recent checkpoint
    ckpt = load_model( args.prefetch_model_folder )


    document_encoder = DocumentEncoder( embed_dim, args.num_heads, args.hidden_dim, vocab_size,  args.max_seq_len, args.max_doc_len , pad_index ,args.n_para_types, unigram_embedding, args.num_enc_layers  )
    if ckpt is not None:
        document_encoder.load_state_dict( ckpt["document_encoder"] )
        LOG("model restored!")
        print("model restored!")

    if args.gpu_list is not None:
        assert len(args.gpu_list) == args.n_device
    else:
        args.gpu_list = np.arange(args.n_device).tolist()
    device = torch.device( "cuda:%d"%(args.gpu_list[0]) if (torch.cuda.is_available() and len(args.gpu_list)>0) else "cpu"  )
    document_encoder = document_encoder.to(device)

    document_encoder_ema = copy.deepcopy( document_encoder ).to(device)

    if device.type == "cuda" and args.n_device > 1:
        document_encoder = nn.DataParallel( document_encoder, args.gpu_list )
        document_encoder_ema = nn.DataParallel( document_encoder_ema, args.gpu_list )
        model_parameters = [ par for par in document_encoder.module.parameters() if par.requires_grad  ] 
    else:
        model_parameters = [ par for par in document_encoder.parameters() if par.requires_grad  ] 

    optimizer = torch.optim.Adam( model_parameters , lr= args.initial_learning_rate,  weight_decay = args.l2_weight  ) 
    
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
    
    # while current_batch < args.max_num_iterations:
    for count in tqdm(range( args.max_num_iterations)):

        current_batch +=1

        batch = train_dataloader.get_next()
        loss = train_iteration(batch)
        running_losses.append(loss)

        update_moving_average(  document_encoder_ema,  document_encoder, args.moving_average_decay)

        if current_batch % args.print_every == 0:
            print("[batch: %05d] loss: %.4f"%( current_batch, np.mean(running_losses) ))
            LOG( "[batch: %05d] loss: %.4f"%( current_batch, np.mean(running_losses) ) )
            os.system( "nvidia-smi > %s/gpu_usage.log"%( args.log_folder ) )
            running_losses = []

        if current_batch % args.save_every == 0 or count == args.max_num_iterations - 1:  
            save_model(  { 
                "current_batch":current_batch,
                "document_encoder": document_encoder_ema,
                "optimizer": optimizer.state_dict()
                } ,  args.prefetch_model_folder+"/model_batch_%d.pt"%( current_batch ), args.max_num_checkpoints )
            print("Model saved!")
            LOG("Model saved!")

