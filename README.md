# Local-Citation-Recommendation
Code for ECIR paper "Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking"

Here we walk thourgh the training, testing and practical usage of the whole prefetching-reranking citation recommendation pipeline on the ACL-200 dataset. 
The same experiments can be conducted on other datasets with proper changes on the configuration files in the config/ folder.

# Hardware Requirement

# Install Dependencies


# Dataset
Download and unzip the ACL-200 dataset. Put the unzipped files to the folder  data/acl/. Inside the data/acl folder it looks like: <br>
 > contexts.json  test.json    papers.json    train.json  val.json

# Prefetching Part
## Training
For simplicity, we refer "MAIN" as the main folder of the repo.

Go to the folder MAIN/src/prefetch/, and run the following command:
```bash
python run_training.py -config_file_path  config/acl/overall_training.config  -model_folder  ../../model/prefetch/acl/
```

This code will automatically handle the loop of  training -> updating paper embeddings -> updating prefetched candidates for contructing triplets -> resuming training.

Parameters:
1. config_file_path: the path to the configuration file for the training. This file contains a list of parameters used during the training loop, including: 
   1) paramaters used for computing paper embeddings (to build the embedding index);
   2) parameters for re-computing prefetched candidates for traiplet mining;
   3) parameters for training;
   4) parameters for valiadting;
2. model_folder, the folder where the checkpoints of models are stored.

In this case, the model checkpoint will be stored in the folder "MAIN/model/prefetch/acl/"; <br>
              the paper embeddings are stored in "MAIN/embedding/prefetch/acl/"; <br>
              the training logfiles are stored in "MAIN/log/prefetch/acl/"; 
              
The file MAIN/log/prefetch/acl/validate_NN.log contains the validation performance of each checkpoint during training. With this information, we can pick up the best-performance model for testing. 

Note: After determining the best checkpoint, removing all the other checkpoints. If there are multiple checkpoints in MAIN/model/prefetch/acl/, the model will use the latest checkpoint by default.

## Testing
To test the performance of the prefetching model, we need to first use the model checkpoint to compute the embedding for each paper in the database. This paper embedding is the index of the paper database, which is then used to perform nearest neighbor search. Then the next step is the test the prefetching performance.

Step 1: Computing paper embeddings
Go to folder  MAIN/src/prefetch/, then
we can compute the paper embedding using one process on a single GPU:
```bash
python compute_papers_embedding.py -config_file_path  config/acl/compute_papers_embedding.config
```

or we can compute embeddings using multiple processes on multiple GPUs. This is useful for large databases:
```bash
bash compute_papers_embedding.sh 20000 10000 config/acl/compute_papers_embedding.config process_synchronization_file_acl_train.temp ../../embedding/prefetch/acl/paper_embedding.pkl_ ../../embedding/prefetch/acl/paper_embedding.pkl
```

Parameters for the bash command:
1. 20000 : The total number of papers in the database. For example in ACL-200 there are 19776 papers, then we can set it to 20000, which is slightly larger than the actual number, but "easier" to do math.
2. 10000 : If we have 2 GPUs, then each GPU only needs to compute 20000/2 = 10000 examples. Similar if we have 8 GPUs then its 2500
3. config/acl/compute_papers_embedding.config : configuration file for a single process
4. process_synchronization_file_acl_train.temp : This file is used for different processes to communicate with each other to make sure all processes end safely. This file can have a any random name, because it will be deleted after the bash script ends.
5. ../../embedding/prefetch/acl/paper_embedding.pkl_ : The prefix of the embedding file save name for each process
6. ../../embedding/prefetch/acl/paper_embedding.pkl : The file name of the paper embedding. This script will automatically merge all embeddings computed from all processes.

Step 2: Test the performance of the model:
```bash
python test.py -config_file_path config/acl/test_NN.config
```
    {'ckpt_name': '../../model/prefetch/acl/model_batch_35000.pt', 'precision': {10: 0.02811684924360981, 20: 0.01866458007303078, 50: 0.010042775169535731, 100: 0.006032342201356285, 200: 0.0035018257694314028, 500: 0.001605633802816902, 1000: 0.00087000521648409, 2000: 0.00046181533646322395}, 'recall': {10: 0.2811684924360981, 20: 0.3732916014606155, 50: 0.5021387584767867, 100: 0.6032342201356286, 200: 0.7003651538862806, 500: 0.8028169014084507, 1000: 0.8700052164840897, 2000: 0.9236306729264476}, 'F1': {10: 0.05112154408095999, 20: 0.03555158109330579, 50: 0.019691716020620618, 100: 0.011945232083854423, 200: 0.006968807503336225, 500: 0.003204857891846127, 1000: 0.0017382721628033805, 2000: 0.0009231690903802587}}


## Use The Prefetcher in Your Python Code

Make sure that paper embeddings have been computed using the final model checkpoint, as discussed above. Then switch the working directory to MAIN/ , and then run the following code:
```python
from citation_recommender import * 
prefetcher = Prefetcher( 
       model_path="model/prefetch/acl/model_batch_35000.pt",
       embedding_path="embedding/prefetch/acl/paper_embedding.pkl",
       gpu_list= [0,1] 
)
```

    embedding loaded


Then we can construct a query, and use the query to find n most relevant papers. The query is a dictionary containing 3 keys: "citing_title","citing_abstract" and "local_context". We can get some real example from the test set, or we can contruct a simple query as follows:


```python
candi_list = prefetcher.get_top_n(
  {
      "citing_title":"",
      "citing_abstract":"",
      "local_context":"machine translation"
  }, 10
)
print(candi_list)
```

    ['N03-1017', 'D09-1023', 'P98-1083', 'P09-1108', 'W12-6208', 'W06-1619', 'P09-2035', 'P07-1037', 'W02-1818', 'P08-1023']


Let's have a look at the titles of top 10 prefetched candidates given our simple query.


```python
papers = json.load( open("data/acl/papers.json","r") )
for item in candi_list:
    print(papers[ item ].get("title",""))
```

    Statistical Phrase-Based Translation
    Feature-Rich Translation by Quasi-Synchronous Lattice Parsing
    Using Decision Trees to Construct a Practical Parser
    K-Best A* Parsing
    WFST-based Grapheme-to-Phoneme Conversion: Open Source Tools for
    Alignment, Model-Building and Decoding
    Extremely Lexicalized Models for Accurate and Fast HPSG Parsing
    Sub-Sentence Division for Tree-Based Machine Translation
    Supertagged Phrase-Based Statistical Machine Translation
    Chinese Base-Phrases Chunking
    Forest-Based Translation


 
