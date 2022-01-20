import subprocess


subprocess.run(  "bash compute_papers_embedding.sh 640000 80000 config/refseer/compute_papers_embedding.config computing_embedding_refseer.temp ../../embedding/prefetch/refseer/paper_embedding.pkl_ ../../embedding/prefetch/refseer/paper_embedding.pkl".split()  )
subprocess.run( "python test.py -config_file_path  config/refseer/test_NN.config  -start 0 -size 10000".split() )


