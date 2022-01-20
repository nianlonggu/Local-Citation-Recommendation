import subprocess


subprocess.run(  "bash compute_papers_embedding.sh 30000 3750 config/dblp/compute_papers_embedding.config computing_embedding_dblp.temp ../../embedding/prefetch/dblp/paper_embedding.pkl_ ../../embedding/prefetch/dblp/paper_embedding.pkl".split()  )
subprocess.run( "python test.py -config_file_path  config/dblp/test_NN.config  -start 0 -size 10000".split() )


