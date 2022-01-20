import subprocess


subprocess.run(  "bash compute_papers_embedding.sh 35000 4375 config/pubmed/compute_papers_embedding.config computing_embedding_pubmed.temp ../../embedding/prefetch/pubmed/paper_embedding.pkl_ ../../embedding/prefetch/pubmed/paper_embedding.pkl".split()  )
subprocess.run( "python test.py -config_file_path  config/pubmed/test_NN.config  -start 0 -size 10000".split() )


