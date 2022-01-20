import subprocess
import os
from glob import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-config_file_path" )
parser.add_argument("-model_folder", default = None)
args= parser.parse_args()

para_list =[line.split("#")[0].strip() for line in open(args.config_file_path,"r").readlines()]

if not os.path.exists( para_list[22] ):
    os.system( "cp "+para_list[21]+" " + para_list[22] )
if not os.path.exists( para_list[24] ):
    os.system( "cp "+para_list[23]+" " + para_list[24] )

if args.model_folder is not None and \
   os.path.exists( args.model_folder ) and \
   len(glob( args.model_folder + "/*.pt" ))>0 :

    print("Found existing model, compute the embedding and get prefetched ids!", flush = True)
    subprocess.run([ "bash","compute_papers_embedding.sh"] + para_list[0:6]   )
    subprocess.run([ "bash","get_prefetched_ids.sh"] + para_list[7:13]   )
    subprocess.run([ "bash","get_prefetched_ids.sh"] + para_list[13:19]   )


sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out_str = sp.communicate()
print(out_str[0].decode("utf-8"), flush = True )

for _ in range( int(para_list[6]) ):
    subprocess.run([ "python","train.py","-config_file_path"] + [para_list[19]]   )
    subprocess.run([ "bash","compute_papers_embedding.sh"] + para_list[0:6]   )
    subprocess.run( "python test.py -config_file_path".split() + [ para_list[20] ] + "-start 0 -size 10000".split() )
    subprocess.run([ "bash","get_prefetched_ids.sh"] + para_list[7:13]   )
    subprocess.run([ "bash","get_prefetched_ids.sh"] + para_list[13:19]   )
    