import subprocess
import os
from glob import glob
import numpy as np
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-config_file_path" )
parser.add_argument("-mode" )
args= parser.parse_args()


config = json.loads(" ".join([ line.split("#")[0] for line in open(args.config_file_path,"r") if line.split("#")[0].strip() != "" ]))
if not os.path.exists( config["train_corpus_path"] ):
    os.system( "cp %s %s"%( config["input_corpus_path_for_get_prefetched_ids_during_training"],
                            config["train_corpus_path"]
                          ) )

assert config["n_device"] > 0
encoder_gpu_list = np.arange( config["n_device"] ).tolist()
ranker_gpu_list = np.arange( config["n_device"] ).tolist()
    
def stringfy( item ):
    if isinstance(item, list):
        return " ".join( [str(el) for el in item] )
    else:
        return str(item)

if args.mode == "compute_paper_embedding":
    subprocess.run(" ".join(list(map(stringfy, 
                        [ "python", "compute_papers_embedding.py",
                          "-unigram_words_path",config["unigram_words_path"],
                          "-prefetch_model_folder",config["prefetch_model_folder"],
                          "-prefetch_embedding_path",config["prefetch_embedding_path"],
                          "-paper_database_path",config["paper_database_path"],
                          "-shuffle", 0,
                          "-size", 0,
                          "-encoder_gpu_list", encoder_gpu_list,
                          "-embed_dim",config["embed_dim"],
                          "-num_heads",config["num_heads"],
                          "-hidden_dim",config["hidden_dim"],
                          "-max_seq_len",config["max_seq_len"],
                          "-max_doc_len",config["max_doc_len"],
                          "-n_para_types",config["n_para_types"],
                          "-num_enc_layers",config["num_enc_layers"],
                          "-document_title_label",config["document_title_label"],
                          "-document_abstract_label",config["document_abstract_label"],
                          "-document_fullbody_label",config["document_fullbody_label"],
                        ]           
                   ))), shell = True
                  )
elif args.mode == "test":
    subprocess.run(" ".join(list(map(stringfy, 
                        [ "python", "test.py",
                          "-log_folder",config["log_folder"],
                          "-log_file_name", config["test_log_file_name"],
                          "-K_list",config["K_list"],
                          "-encoder_gpu_list", encoder_gpu_list,
                          "-ranker_gpu_list", ranker_gpu_list,
                          "-input_corpus_path", config["input_corpus_path_for_test"],
                          "-unigram_words_path",config["unigram_words_path"],
                          "-prefetch_model_folder",config["prefetch_model_folder"],
                          "-prefetch_embedding_path",config["prefetch_embedding_path"],
                          "-paper_database_path",config["paper_database_path"],
                          "-context_database_path",config["context_database_path"],
                          "-embed_dim",config["embed_dim"],
                          "-num_heads",config["num_heads"],
                          "-hidden_dim",config["hidden_dim"],
                          "-max_seq_len",config["max_seq_len"],
                          "-max_doc_len",config["max_doc_len"],
                          "-n_para_types",config["n_para_types"],
                          "-num_enc_layers",config["num_enc_layers"],
                          "-citation_title_label",config["citation_title_label"],
                          "-citation_abstract_label",config["citation_abstract_label"],
                          "-citation_context_label",config["citation_context_label"],
                        ]           
                   ))), shell = True
                  )
elif args.mode == "get_training_examples_with_prefetched_ids_for_reranking":
    subprocess.run(" ".join(list(map(stringfy, 
                        [ "python", "get_prefetched_ids.py",
                          "-shuffle", 1,
                          "-size", config["num_training_examples_with_prefetched_ids_for_reranking"],
                          "-top_K", config["top_K_prefetched_ids_for_reranking"],
                          "-encoder_gpu_list", encoder_gpu_list,
                          "-ranker_gpu_list", ranker_gpu_list,
                          "-input_corpus_path", config["input_corpus_path_for_get_prefetched_ids_during_training"],
                          "-output_corpus_path", config["output_corpus_path_for_get_prefetched_ids_during_training"],
                          "-unigram_words_path",config["unigram_words_path"],
                          "-prefetch_model_folder",config["prefetch_model_folder"],
                          "-prefetch_embedding_path",config["prefetch_embedding_path"],
                          "-paper_database_path",config["paper_database_path"],
                          "-context_database_path",config["context_database_path"],
                          "-embed_dim",config["embed_dim"],
                          "-num_heads",config["num_heads"],
                          "-hidden_dim",config["hidden_dim"],
                          "-max_seq_len",config["max_seq_len"],
                          "-max_doc_len",config["max_doc_len"],
                          "-n_para_types",config["n_para_types"],
                          "-num_enc_layers",config["num_enc_layers"],
                          "-citation_title_label",config["citation_title_label"],
                          "-citation_abstract_label",config["citation_abstract_label"],
                          "-citation_context_label",config["citation_context_label"],
                        ]           
                   ))), shell = True
                  )
    
elif args.mode == "get_val_examples_with_prefetched_ids_for_reranking":
    subprocess.run(" ".join(list(map(stringfy, 
                        [ "python", "get_prefetched_ids.py",
                          "-shuffle", 1,
                          "-size", config["num_val_examples_with_prefetched_ids_for_reranking"],
                          "-top_K", config["top_K_prefetched_ids_for_reranking"],
                          "-encoder_gpu_list", encoder_gpu_list,
                          "-ranker_gpu_list", ranker_gpu_list,
                          "-input_corpus_path", config["input_corpus_path_for_validation"],
                          "-output_corpus_path", config["output_corpus_path_for_validation_with_prefetched_ids"],
                          "-unigram_words_path",config["unigram_words_path"],
                          "-prefetch_model_folder",config["prefetch_model_folder"],
                          "-prefetch_embedding_path",config["prefetch_embedding_path"],
                          "-paper_database_path",config["paper_database_path"],
                          "-context_database_path",config["context_database_path"],
                          "-embed_dim",config["embed_dim"],
                          "-num_heads",config["num_heads"],
                          "-hidden_dim",config["hidden_dim"],
                          "-max_seq_len",config["max_seq_len"],
                          "-max_doc_len",config["max_doc_len"],
                          "-n_para_types",config["n_para_types"],
                          "-num_enc_layers",config["num_enc_layers"],
                          "-citation_title_label",config["citation_title_label"],
                          "-citation_abstract_label",config["citation_abstract_label"],
                          "-citation_context_label",config["citation_context_label"],
                        ]           
                   ))), shell = True
                  )
    
elif args.mode == "train":
    for loop_count in range( config["max_num_loops_for_training_updating_embedding_and_prefetched_ids"] ):
        subprocess.run(" ".join(list(map(stringfy, 
                        [ "python","train.py",
                          "-unigram_words_path", config["unigram_words_path"],
                          "-unigram_embedding_path", config["unigram_embedding_path"],
                          "-prefetch_model_folder", config["prefetch_model_folder"],
                          "-paper_database_path", config["paper_database_path"],
                          "-context_database_path", config["context_database_path"],
                          "-train_corpus_path", config["train_corpus_path"],
                          "-log_folder", config["log_folder"],
                          "-log_file_name", config["train_log_file_name"],
                          "-max_num_samples_per_batch", config["max_num_samples_per_batch"],
                          "-n_device", config["n_device"],
                          "-print_every", config["print_every"],
                          "-save_every", config["save_every"],
                          "-max_num_iterations",config["max_num_iterations"],
                          "-max_n_positive",config["max_n_positive"],
                          "-max_n_hard_negative",config["max_n_hard_negative"],
                          "-max_n_easy_negative",config["max_n_easy_negative"],
                          "-num_workers",config["num_workers"],
                          "-initial_learning_rate",config["initial_learning_rate"],
                          "-l2_weight",config["l2_weight"],
                          "-dropout_rate",config["dropout_rate"],
                          "-moving_average_decay",config["moving_average_decay"],
                          "-base_margin",config["base_margin"],
                          "-similarity",config["similarity"],
                          "-positive_irrelevance_levels",config["positive_irrelevance_levels"],
                          "-max_num_checkpoints",config["max_num_checkpoints"],
                          "-embed_dim",config["embed_dim"],
                          "-num_heads",config["num_heads"],
                          "-hidden_dim",config["hidden_dim"],
                          "-max_seq_len",config["max_seq_len"],
                          "-max_doc_len",config["max_doc_len"],
                          "-n_para_types",config["n_para_types"],
                          "-num_enc_layers",config["num_enc_layers"],
                          "-document_title_label",config["document_title_label"],
                          "-document_abstract_label",config["document_abstract_label"],
                          "-document_fullbody_label",config["document_fullbody_label"],
                          "-citation_title_label",config["citation_title_label"],
                          "-citation_abstract_label",config["citation_abstract_label"],
                          "-citation_fullbody_label",config["citation_fullbody_label"],
                          "-citation_context_label",config["citation_context_label"],
                          "-padding_paragraph_label",config["padding_paragraph_label"],
                        ]
                    ))), shell = True
                  )
    
        subprocess.run(" ".join(list(map(stringfy, 
                        [ "python", "compute_papers_embedding.py",
                          "-unigram_words_path",config["unigram_words_path"],
                          "-prefetch_model_folder",config["prefetch_model_folder"],
                          "-prefetch_embedding_path",config["prefetch_embedding_path"],
                          "-paper_database_path",config["paper_database_path"],
                          "-shuffle", 1,
                          "-size", config["num_papers_with_updated_embeddings_per_loop"],
                          "-encoder_gpu_list", encoder_gpu_list,
                          "-embed_dim",config["embed_dim"],
                          "-num_heads",config["num_heads"],
                          "-hidden_dim",config["hidden_dim"],
                          "-max_seq_len",config["max_seq_len"],
                          "-max_doc_len",config["max_doc_len"],
                          "-n_para_types",config["n_para_types"],
                          "-num_enc_layers",config["num_enc_layers"],
                          "-document_title_label",config["document_title_label"],
                          "-document_abstract_label",config["document_abstract_label"],
                          "-document_fullbody_label",config["document_fullbody_label"],
                        ]           
                   ))), shell = True
                  )
    
        subprocess.run(" ".join(list(map(stringfy, 
                        [ "python", "get_prefetched_ids.py",
                          "-shuffle", 1,
                          "-size", config["num_training_examples_with_updated_prefetched_ids_per_loop"],
                          "-top_K", config["top_K_prefetched_ids_for_mining_hard_negative"],
                          "-encoder_gpu_list", encoder_gpu_list,
                          "-ranker_gpu_list", ranker_gpu_list,
                          "-input_corpus_path", config["input_corpus_path_for_get_prefetched_ids_during_training"],
                          "-output_corpus_path", config["output_corpus_path_for_get_prefetched_ids_during_training"],
                          "-unigram_words_path",config["unigram_words_path"],
                          "-prefetch_model_folder",config["prefetch_model_folder"],
                          "-prefetch_embedding_path",config["prefetch_embedding_path"],
                          "-paper_database_path",config["paper_database_path"],
                          "-context_database_path",config["context_database_path"],
                          "-embed_dim",config["embed_dim"],
                          "-num_heads",config["num_heads"],
                          "-hidden_dim",config["hidden_dim"],
                          "-max_seq_len",config["max_seq_len"],
                          "-max_doc_len",config["max_doc_len"],
                          "-n_para_types",config["n_para_types"],
                          "-num_enc_layers",config["num_enc_layers"],
                          "-citation_title_label",config["citation_title_label"],
                          "-citation_abstract_label",config["citation_abstract_label"],
                          "-citation_context_label",config["citation_context_label"],
                        ]           
                   ))), shell = True
                  )
    
            
        subprocess.run(" ".join(list(map(stringfy, 
                        [ "python", "test.py",
                          "-log_folder",config["log_folder"],
                          "-log_file_name", config["val_log_file_name"],
                          "-size", config["num_val_examples_per_loop"],
                          "-K_list",config["K_list"],
                          "-encoder_gpu_list", encoder_gpu_list,
                          "-ranker_gpu_list", ranker_gpu_list,
                          "-input_corpus_path", config["input_corpus_path_for_validation"],
                          "-unigram_words_path",config["unigram_words_path"],
                          "-prefetch_model_folder",config["prefetch_model_folder"],
                          "-prefetch_embedding_path",config["prefetch_embedding_path"],
                          "-paper_database_path",config["paper_database_path"],
                          "-context_database_path",config["context_database_path"],
                          "-embed_dim",config["embed_dim"],
                          "-num_heads",config["num_heads"],
                          "-hidden_dim",config["hidden_dim"],
                          "-max_seq_len",config["max_seq_len"],
                          "-max_doc_len",config["max_doc_len"],
                          "-n_para_types",config["n_para_types"],
                          "-num_enc_layers",config["num_enc_layers"],
                          "-citation_title_label",config["citation_title_label"],
                          "-citation_abstract_label",config["citation_abstract_label"],
                          "-citation_context_label",config["citation_context_label"],
                        ]           
                   ))), shell = True
                  )