from rankers import *
from utils import *
import os
import pickle
from glob import glob
import numpy as np
from array import array
import argparse
import json
from tqdm import tqdm


def add_unigram( inv_idx, index, doc_unigrams ):
    unique_unigrams = {}
    for unigram in doc_unigrams:
        unique_unigrams[unigram] = unique_unigrams.get( unigram, 0 ) + 1
            
    for unigram in unique_unigrams:
        if unigram not in inv_idx:
            inv_idx[ unigram ] = { "doc_indices": array( "I", [index] ), "term_frequencies": array( "I", [ unique_unigrams[unigram] ] )   }
        else:
            inv_idx[ unigram ][ "doc_indices" ].append( index )
            inv_idx[ unigram ][ "term_frequencies" ].append( unique_unigrams[unigram] )


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file_path" )
    args_input = parser.parse_args()
    args = Dict2Class(json.load(open(args_input.config_file_path)))

    paper_database = json.load(open(args.paper_database_path))
    

    if args.cr_mode == "local":
        #available_paper_ids = list( paper_database.keys() )
        available_paper_ids = []
        for pid in paper_database:
            if paper_database.get( "is_in_paper_database", True ):
                available_paper_ids.append(pid)
    elif args.cr_mode == "global":
        available_paper_ids = json.load(open( args.available_paper_ids_path))


    inv_idx = {}
    index_to_id_mapper = {}
    id_to_index_mapper = {}
    index_to_doc_length_mapper = {}
    sent_tok = SentenceTokenizer()

    for index, paperid in enumerate( tqdm( available_paper_ids ) ):
        paper = paper_database[paperid]

        index_to_id_mapper[index] = paperid
        id_to_index_mapper[paperid] = index

        paper_words =sent_tok.tokenize( paper["title"]+" " + paper["abstract"]).split()
        add_unigram( inv_idx, index, paper_words )
        index_to_doc_length_mapper[index] = len(paper_words)

    for unigram in inv_idx:
        inv_idx[unigram]["doc_indices"] = np.array( inv_idx[unigram]["doc_indices"]  )
        inv_idx[unigram]["term_frequencies"] = np.array( inv_idx[unigram]["term_frequencies"] )

    with open(args.inverted_index_path, "wb" ) as f:
        pickle.dump( { "id_to_index_mapper": id_to_index_mapper,
               "index_to_doc_length_mapper": index_to_doc_length_mapper,
               "num_of_docs": index+1, 
               "index_to_id_mapper": index_to_id_mapper, 
               "inv_idx": inv_idx }, f, -1 )




    

    
    
    

