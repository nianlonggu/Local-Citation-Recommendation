# Local-Citation-Recommendation
Code for ECIR 2022 paper Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking

# Update 07-11-2022
1. Cleaned the code for training and testing the prefetching system, to make it easier to read and to run.
2. Simplify the information in config file, now there is only one global configuration file for prefetching and it is more readable.
3. Optimize the GPU usage, now the system can be trained using a single GPU.
4. Introduced the structure of the dataset and showed how to build your custom dataset and train a citation recommendation system on that.
5. Provided a step-by-step tutorial on google colab, illustrating the whole process of training and testing of the entire prefetching and reranking system. 


# Hardware Requirement
1. OS: Ubuntu 20.04 or 18.04
2. >= 1 GPU (12 GB RAM)


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
gdown<br>
transformers<br>
2) Install the following package with conda:
cupy:
```bash
conda install -c conda-forge cupy
```
pytorch (pytorch 1.9.0 or higher):
```bash
conda install -c conda-forge pytorch-gpu
```
3) Download nltk data. Run the following Python code in the anaconda environment:
```python
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
```
# Download Glove Embedding 
For simplicity, we refer **MAIN** as the main folder of the repo.

```bash
cd MAIN; gdown  https://drive.google.com/uc?id=1T2R1H8UstSILH_JprUaPNY0fasxD2Txr; unzip model.zip; rm model.zip
```

# Prepare Dataset

## Build your custom dataset

The custom dataset will contain 5 components: contexts, papers, training/validation/test sets

```python
import json
contexts = json.load(open("data/custom/contexts.json"))
papers = json.load(open("data/custom/papers.json"))
train_set = json.load(open("data/custom/train.json"))
val_set = json.load(open("data/custom/val.json"))
test_set = json.load(open("data/custom/test.json"))
```

### contexts contain the local contexts that cite a paper.


```python
for key in contexts:
    break
contexts[key]
```

    {'masked_text': ' retaining all stopwords. These measures have been shown to correlate best with human judgments in general, but among the automatic measures, ROUGE-1 and ROUGE-2 also correlate best with the Pyramid ( TARGETCIT ; OTHERCIT) and Responsiveness manual metrics (OTHERCIT). Moreover, ROUGE-1 has been shown to best reflect human-automatic summary comparisons (OTHERCIT). For single concept systems, the results are s',
     'context_id': 'P15-2138_N04-1019_0',
     'citing_id': 'P15-2138',
     'refid': 'N04-1019'}


### papers contain the papers database, each paper has title and abstract.

```python
for key in papers:
    break
papers[key]
```
    {'title': 'Shared task system description:\nFrustratingly hard compositionality prediction',
     'abstract': 'We considered a wide range of features for the DiSCo 2011 shared task about compositionality prediction for word pairs, including COALS-based endocentricity scores, compositionality scores based on distributional clusters, statistics about wordnet-induced paraphrases, hyphenation, and the likelihood of long translation equivalents in other languages. Many of the features we considered correlated significantly with human compositionality scores, but in support vector regression experiments we obtained the best results using only COALS-based endocentricity scores. Our system was nevertheless the best performing system in the shared task, and average error reductions over a simple baseline in cross-validation were 13.7% for English and 50.1% for German.'}


### train/val/test set contain the context_id (used for get the local context information and cited and citing papers information)


```python
train_set[0]
```
    {'context_id': 'P12-1066_D10-1120_1', 'positive_ids': ['D10-1120']}

positive_ids means the paper that is actually cited by the context. In this experiment the positive_ids always has one paper.

You can create you own dataset with the same structure, and then train the citation recommendation system. We provide a easy tutorial on how to train the prefetching and reranking system step by step. The tutorial is available at google colab:

## Download Processed Dataset
You can download our processed dataset from [Google Drive](https://drive.google.com/drive/folders/1QwQuJsBOGEESFTgl-7wWbqcig7vJNlQ2?usp=sharing)
(There can be some additional information in the processed dataset other than what have been displayed in the examples above. They are irrelevant information.)


# Prefetching Part
In the following experiment, we use the "custom" dataset as an example. This dataset is the same as the ACL dataset. If you have created you dataset, you need to modify the config file at MAIN/src.prefetch/config/YOUR_DATASET_NAME/global_config.cfg accordingly.
## Training

Go to the folder MAIN/src/prefetch/, and run the following command:
```bash
python run_overall_training.py -config_file_path config/custom/global_config.cfg
```

This code will automatically handle the loop of  training -> updating paper embeddings -> updating prefetched candidates for contructing triplets -> resuming training.


In this case, the model checkpoint will be stored in the folder "MAIN/model/prefetch/custom/"; <br>
              the paper embeddings are stored in "MAIN/embedding/prefetch/custom/"; <br>
              the training log files are stored in "MAIN/log/prefetch/custom/"; 
              
The file MAIN/log/prefetch/custom/validate_NN.log contains the validation performance of each checkpoint during training. With this information, we can pick up the best-performance model for testing. 

You can specify where to store the checkpoint, log files and other parammeters by modifying the config/custom/global_config.cfg configuration file.

Note: **Before testing, after determining the best checkpoint, removing all the other checkpoints. If there are multiple checkpoints in MAIN/model/prefetch/custom/, the model will use the latest checkpoint by default.**


## Testing
To test the performance of the prefetching model, we need to first use the model checkpoint to compute the embedding for each paper in the database. This paper embedding is the index of the paper database, which is then used to perform nearest neighbor search. Then the next step is the test the prefetching performance.

Go to folder  MAIN/src/prefetch/, then open Python environment, and run the following Python Code:
```python
import subprocess
import os
from glob import glob
import numpy as np
import json


config_file_path = "config/custom/global_config.cfg"
config = json.loads(" ".join([ line.split("#")[0] for line in open(config_file_path,"r") if line.split("#")[0].strip() != "" ]))
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
    
subprocess.run(" ".join(list(map(stringfy, 
                        [ "python", "compute_papers_embedding.py",
                          "-unigram_words_path",config["unigram_words_path"],
                          "-prefetch_model_folder",config["prefetch_model_folder"],
                          "-prefetch_embedding_path",config["prefetch_embedding_path"],
                          "-paper_database_path",config["paper_database_path"],
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

```



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
@InProceedings{10.1007/978-3-030-99736-6_19,
author="Gu, Nianlong
and Gao, Yingqiang
and Hahnloser, Richard H. R.",
editor="Hagen, Matthias
and Verberne, Suzan
and Macdonald, Craig
and Seifert, Christin
and Balog, Krisztian
and N{\o}rv{\aa}g, Kjetil
and Setty, Vinay",
title="Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-Based Reranking",
booktitle="Advances in Information Retrieval",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="274--288",
abstract="The goal of local citation recommendation is to recommend a missing reference from the local citation context and optionally also from the global context. To balance the tradeoff between speed and accuracy of citation recommendation in the context of a large-scale paper database, a viable approach is to first prefetch a limited number of relevant documents using efficient ranking methods and then to perform a fine-grained reranking using more sophisticated models. In that vein, BM25 has been found to be a tough-to-beat approach to prefetching, which is why recent work has focused mainly on the reranking step. Even so, we explore prefetching with nearest neighbor search among text embeddings constructed by a hierarchical attention network. When coupled with a SciBERT reranker fine-tuned on local citation recommendation tasks, our hierarchical Attention encoder (HAtten) achieves high prefetch recall for a given number of candidates to be reranked. Consequently, our reranker requires fewer prefetch candidates to rerank, yet still achieves state-of-the-art performance on various local citation recommendation datasets such as ACL-200, FullTextPeerRead, RefSeer, and arXiv.",
isbn="978-3-030-99736-6"
}


```
