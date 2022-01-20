import pickle
import os
import numpy as np
import time

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-num_process", type = int, default = None)
parser.add_argument("-signal_file", default = None)

parser.add_argument("-slice_name_prefix")
parser.add_argument("-save_name")

args = parser.parse_args( )

num_process = args.num_process
signal_file = args.signal_file
slice_name_prefix = args.slice_name_prefix
save_name = args.save_name

## waiting util the embeddings of each slice have been computed

if args.num_process is not None:
  while True:
    if not os.path.exists(signal_file):
      time.sleep(1)
      continue
    if len(open(signal_file,"r").readlines()) == num_process:
      break
    time.sleep(1)


data_folder = "/".join(slice_name_prefix.split("/")[:-1])+"/"
prefix = slice_name_prefix.split("/")[-1]
flist = [data_folder+"/"+f for f in os.listdir(data_folder) if f.startswith(prefix)  ]
flist.sort(key = lambda x: int( x.split("_")[-2] )  )


if len(flist)>0:
    id_to_index_mapper = {}
    index_to_id_mapper = {}
    embedding =[]   

    for fname in flist:
        with open( fname , "rb" ) as f:
            info = pickle.load(f)
            id_to_index_mapper.update(info["id_to_index_mapper"])
            index_to_id_mapper.update(info["index_to_id_mapper"])
            embedding.append(info["embedding"]  )
        os.remove( fname )    

    embedding = np.concatenate(embedding, axis = 0)   

    with open(save_name, "wb") as f:
        pickle.dump( { 
          "id_to_index_mapper":id_to_index_mapper,
          "index_to_id_mapper":index_to_id_mapper,
          "embedding": embedding
        }, f, -1  )
else:
    print("No files with the name prefix found!")



