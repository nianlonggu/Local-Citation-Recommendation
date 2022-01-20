import os
import json
import time
from tqdm import tqdm

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-slice_name_prefix" )
    parser.add_argument("-save_name" )

    parser.add_argument("-num_process", type = int)
    parser.add_argument("-signal_file")
    
    args = parser.parse_args()

    ## waiting util the all processes have finished
    while True:
        if not os.path.exists(args.signal_file):
            time.sleep(1)
            continue
        if len(open(args.signal_file,"r").readlines()) == args.num_process:
            break
        time.sleep(1)


    folder = "/".join(args.slice_name_prefix.split("/")[:-1])+"/"
    prefix = args.slice_name_prefix.split("/")[-1]


    flist = [ folder +"/"+ fname  for fname in os.listdir(folder)  if fname.startswith( prefix ) ]
    flist.sort( key = lambda x: int( x.split("_")[-2] )  )

    count = 0
    fw = open( args.save_name , "w" )
    for fname in flist:
        print(fname)
        with open(fname,"r") as f:
            for line in tqdm(f):
                fw.write(line)
                count +=1
        
        os.remove( fname )

    print("Number of examples:", count  )

    fw.close()
