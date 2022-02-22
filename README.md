# Local-Citation-Recommendation
Code for ECIR 2022 paper "Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking"

Here we walk thourgh the training, testing and practical usage of the whole prefetching-reranking citation recommendation pipeline on the ACL-200 dataset. 
The same experiments can be conducted on other datasets with proper changes on the configuration files in the config/ folder.

# Hardware Requirement
1. OS: Ubuntu 20.04 or 18.04
2. If you want to train the prefetcher and reranker from scratch, you can use:
    1) 2 x nVidia RTX A6000 48GB GPUs, or
    2) 8 x GTX 1080Ti (11GB) GPUs

  In other words, the total GPU RAM should be around 90 GB. We need a large RAM because we need to do within-minibatch traiplet mining so we used a large batch size (around 512). <br>
  Note that the training example below uses 2 GPUs by default. If you want to use 8 GPUs, you need to modify the configuration files in the folder src/prefetch/config/ and src/rerank/config. The modification includes 1) changing the number of examples per gpu process in the overall_training.config, and 2) changing the value of "n_device" from 2 to 8.
  
3. If you want to run the chapter **Use The Prefetcher in Python Code** and the chapter **Use The Reranker in Python Code**, a single GPU should be enough.


# Install Dependencies
1. Install anaconda;
2. Create an anaconda environment (environment name: lcr, python version=3.7):
```bash
conda create -n lcr python=3.7 -y
```
3. Activate the environment:
```bash
source activate lcr
```
4. Install requirements (after activate the anaconda environment):
1) Install the following package with pip:<br>
numpy<br>
tqdm<br>
matplotlib<br>
nltk<br>
2) Install the following package with conda:
cupy:
```bash
conda install -c conda-forge cupy
```
pytorch (pytorch 1.9.0):
```bash
conda install -c conda-forge pytorch-gpu
```
3) Install transformers via:
```bash
pip install transformers
```
4) Download nltk data. Run the following Python code in the anaconda environment:
```python
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
```





# Download Dataset, Embedding and Vocabulary
For simplicity, we refer **MAIN** as the main folder of the repo.

Downloading link to all the data:  [DATA Link](https://drive.google.com/drive/folders/1hTgzScKbY2te9r6UfXJeSAPlPnekPLx9?usp=sharing)
 (This experiment is demoed using the ACL-200 dataset.)

This link contains all the data (dataset, embedding, vocabulary and model) for the experiments. After downloading and unzipping, we can use part of them based on different usage scenarios:
1. Among them, the ACL-200 dataset, glove embedding, glove vocabulary are necessary for all experiments.
2. The pretrained prefetching model on ACL-200 is only needed if you want to directly run the code in chapter **Use The Prefetcher in Python Code** without training from scratch;
3. The pretrained reranking model on ACL-200 is only needed if you want to directly run the code in chapter **Use The Reranker in Python Code** without training from scratch.


## Dataset
Download and unzip the ACL-200 dataset. Put the unzipped files to the folder  MAIN/data/acl/<br> 
Inside the data/acl folder it looks like: <br>
 > contexts.json  test.json    papers.json    train.json  val.json test_with_oracle_prefetched_ids_for_reranking.json
## Embedding, Vocabulary 
Put the glove embedding  "unigram_embeddings_200dim.pkl" and vocabulary  "vocabulary_200dim.pkl" to the folder MAIN/model/glove/

# Prefetching Part
## Training

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


## Use The Prefetcher in Python Code

Before runing this code, make sure the following steps have been done in the same order:
1. The dataset for the ACL-200 have been downloaded from [DATA Link](https://drive.google.com/drive/folders/1hTgzScKbY2te9r6UfXJeSAPlPnekPLx9?usp=sharing), and placed at the correct position MAIN/data/acl/
2. The Glove embedding and vocabulary have been downloaded from [DATA Link](https://drive.google.com/drive/folders/1hTgzScKbY2te9r6UfXJeSAPlPnekPLx9?usp=sharing), and placed at the position MAIN/model/glove/. Inside MAIN/model/glove/ there will be two files: unigram_embeddings_200dim.pkl, vocabulary_200dim.pkl
3. The best checkpoint of the prefetcher has been trained, selected, and placed at MAIN/model/prefetch/acl/, or you can download the checkpoint [DATA Link](https://drive.google.com/drive/folders/1hTgzScKbY2te9r6UfXJeSAPlPnekPLx9?usp=sharing) and place it at MAIN/model/prefetch/acl/.
4. The paper embeddings have been computed using the embedding computation script shown above.

Then switch the working directory to MAIN/ , and then run the following code:
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


# Reranking Part
## Training

In order to train the reranker, we need to first create the training dataset. More specifically, for each query in the training set, we first use the trained HAtten prefetcher to prefetch a list of (2000) candidates. Then within the 2000 prefetched candidates we can construct triplets to train the reranker.

Step 1: create the training and validating set. Note: Before runing this command, make sure that the best checkpoint of the prefetcher has been stored in MAIN/model/prefetch/acl/, and the paper embeddings has been computed using the best checkpoint!
```bash
bash get_prefetched_ids_for_reranking.sh 32000 16000 config/acl/get_NN_prefetched_ids_for_reranking_train.config   get_prefetched_for_reranking_acl_scibert.signal ../../data/acl/train_with_NN_prefetched_ids_for_reranking_scibert.json_  ../../data/acl/train_with_NN_prefetched_ids_for_reranking_scibert.json
```
Illustration of parameters:
1. 32000 : The number of training examples for which we need to obtain the prefetched candidates. ACL-200 has 30390 examples, we set 32000 because it is easier to do math. Note: for some large datasets such as RefSeer and arXiv, we might not need the whole training dataset to train the reranker. In that case, we can set any reasonable number smaller than the actual number of total training examples, which means we can train the reranker only using a subset of examples.
2. 16000 : The number of training examples allocated per gpu. Here we use 2 GPUs so the value is 32000/2 = 16000
3. config/acl/get_NN_prefetched_ids_for_reranking_train.config : configuration file for a single process
4. get_prefetched_for_reranking_acl_scibert.signal : a random file used for synchronization between multiple processes
5. ../../data/acl/train_with_NN_prefetched_ids_for_reranking_scibert.json_ : prefix of the reranker training dataset generated by one process
6. ../../data/acl/train_with_NN_prefetched_ids_for_reranking_scibert.json : final name of the training set.

Similarly we create the validating set:
```bash
bash get_prefetched_ids_for_reranking.sh 10000 5000 config/acl/get_NN_prefetched_ids_for_reranking_validate.config   get_prefetched_for_reranking_acl_scibert.signal ../../data/acl/val_with_NN_prefetched_ids_for_reranking_scibert.json_  ../../data/acl/val_with_NN_prefetched_ids_for_reranking_scibert.json
```

Step 2: Start the training:
```bash
python train.py -config_file_path  config/acl/scibert/training_NN_prefetch.config
```

The logfiles are stored at MAIN/log/rerank/acl/scibert/NN_prefetch/ . By checking the validation loss in the train.log file, we can select the best-performing checkpoint, and use it for testing.

## Testing
Note: make sure that in the model folder MAIN/model/rerank/acl/scibert/NN_prefetch/ there is only one checkpoint of the best model. Otherwise the latest model is used by default.

```bash
python test.py -config_file_path config/acl/scibert/testing_oracle_prefetch.config
```

## Use The Reranker in Python Code

Before runing this code, make sure the following steps have been done in the same order:
1. The dataset for the ACL-200 have been downloaded from [DATA Link](https://drive.google.com/drive/folders/1hTgzScKbY2te9r6UfXJeSAPlPnekPLx9?usp=sharing), and placed at the correct position MAIN/data/acl/
2. The best checkpoint of the reranker has been trained, selected, and placed at MAIN/model/prefetch/acl/, or you can download the checkpoint [DATA Link](https://drive.google.com/drive/folders/1hTgzScKbY2te9r6UfXJeSAPlPnekPLx9?usp=sharing) and place it at MAIN/model/rerank/acl/scibert/NN_prefetch/

Then switch the working directory to MAIN/ , and then run the following code:

```python
from citation_recommender import *
## if no GPU, then set gpu_list as []
reranker = Reranker( model_path = "model/rerank/acl/scibert/NN_prefetch/model_batch_91170.pt", gpu_list = [0,1] )
```

Here we get one test example from the test set for reranking. In the test set, for each query, the prefetched candidates are feteched by a oracle-BM25 as descibed in the paper. In other words, if the actually cited paper does not exist in the top 2000 candidates, then manually insert it.


```python
papers = json.load(open("data/acl/papers.json"))
contexts = json.load(open("data/acl/contexts.json"))
test_rerank = []
with open("data/acl/test_with_oracle_prefetched_ids_for_reranking.json") as f:
    for line in f:
        test_rerank.append(json.loads(line))
test_example = test_rerank[0]
```

Then we can construct the parameters used for reranking.

The reranker takes the following parameters:
1. citing_title: the title of the citing paper;
2. citing_abstract: the abstract of the citing paper;  Both 1 and 2 compose the global citation context
3. local context: the local citation sentence
4. candidate_list: a list of prefetched candidates. Each element is a dictionary that contains the "paper_id", "title" and "abstract" of a candidate paper.


```python
citing_paper = papers[contexts[test_example["context_id"]]["citing_id"]]
citing_title = citing_paper.get("title","")
citing_abstract = citing_paper.get("abstract","")
local_context = contexts[test_example["context_id"]]["masked_text"]
candidate_list = [  {"paper_id": pid,
                     "title":papers[pid].get("title",""),
                     "abstract":papers[pid].get("abstract","")}
                            for pid in test_example["prefetched_ids"] ] 
# start reranking
reranked_candidate_list = reranker.rerank( citing_title,citing_abstract,local_context, candidate_list )
```

We can have a check if the reranker works by checking if the actually cited paper is closer to the top position after reranking.


```python
positive_id  = test_example["positive_ids"][0]
for count, candidate in enumerate(candidate_list):
    if positive_id == candidate["paper_id"]:
        break
print("Berfore reranking, the position of the actually cited paper:",count+1)

positive_id  = test_example["positive_ids"][0]
for count, candidate in enumerate(reranked_candidate_list):
    if positive_id == candidate["paper_id"]:
        break
print("After reranking, the position of the actually cited paper:",count+1)
```

    Berfore reranking, the position of the actually cited paper: 2000
    After reranking, the position of the actually cited paper: 3


So the reranker clearly helps.


# References

When using our code or models for your application, please cite the following paper:

```
@article{gu2021local,
  title={Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking},
  author={Gu, Nianlong and Gao, Yingqiang and Hahnloser, Richard HR},
  journal={arXiv preprint arXiv:2112.01206},
  year={2021}
}
```
