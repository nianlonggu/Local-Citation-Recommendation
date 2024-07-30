[![DOI](https://img.shields.io/badge/DOI-10%2E1007%2F978--3--030--99736--6__19-blue)](https://doi.org/10.1007/978-3-030-99736-6_19)
# Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking

<a href="https://colab.research.google.com/github/nianlonggu/Local-Citation-Recommendation/blob/main/Turorial_Local_Citation_Recommendation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Code for ECIR 2022 paper [Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking](https://link.springer.com/chapter/10.1007/978-3-030-99736-6_19)

## Update 07-11-2022
1. Cleaned the code for training and testing the prefetching system, to make it easier to read and to run.
2. Simplify the information in config file, now there is only one global configuration file for prefetching and it is more readable.
3. Optimize the GPU usage, now the system can be trained using a single GPU.
4. Introduced the structure of the dataset and showed how to build your custom dataset and train a citation recommendation system on that.
5. Provided a step-by-step tutorial on google colab, illustrating the whole process of training and testing of the entire prefetching and reranking system. 
https://github.com/nianlonggu/Local-Citation-Recommendation/blob/main/Turorial_Local_Citation_Recommendation.ipynb
Try it here
<a href="https://colab.research.google.com/github/nianlonggu/Local-Citation-Recommendation/blob/main/Turorial_Local_Citation_Recommendation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Hardware Requirement
1. OS: Ubuntu 20.04 or 18.04
2. 1 or more GPUs

## Install Dependencies

Create anaconda environment
```bash
conda create -n hatten python=3.10 -y
```

Activate the anaconda environment
```bash
conda activate hatten
```
Install dependencies
```bash
pip install -r requirements.txt
```
Install CUPY GPU, based on your GUDA version:

For CUDA 11.x:
```bash
pip install cupy-cuda11x 
```
For CUDA 12.x:
```bash
pip install cupy-cuda12x 
```
Activate Python environment and download nltk data:

```python
import nltk
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)
```

## Download Glove Embedding 
For simplicity, we refer **MAIN** as the main folder of the repo.


```python
!gdown  https://drive.google.com/uc?id=1T2R1H8UstSILH_JprUaPNY0fasxD2Txr; unzip model.zip; rm model.zip
```

    Downloading...
    From: https://drive.google.com/uc?id=1T2R1H8UstSILH_JprUaPNY0fasxD2Txr
    To: /content/Local-Citation-Recommendation/model.zip
    100% 295M/295M [00:00<00:00, 300MB/s]
    Archive:  model.zip
       creating: model/
       creating: model/glove/
      inflating: model/glove/unigram_embeddings_200dim.pkl  
      inflating: model/glove/vocabulary_200dim.pkl  


## Prepare Dataset

### Option 1: Build your custom dataset 
**This github repo contains a "pseudo" custom dataset that is actually ACL-200**

The custom dataset will contain 5 components: contexts, papers, training/validation/test sets


```python
import json
contexts = json.load(open("data/custom/contexts.json"))
papers = json.load(open("data/custom/papers.json"))
train_set = json.load(open("data/custom/train.json"))
val_set = json.load(open("data/custom/val.json"))
test_set = json.load(open("data/custom/test.json"))
```

#### contexts contain the local contexts that cite a paper.


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



#### train/val/test set contain the context_id (used for get the local context information and cited and citing papers information)


```python
train_set[0]
```




    {'context_id': 'P12-1066_D10-1120_1', 'positive_ids': ['D10-1120']}



positive_ids means the paper that is actually cited by the context. In this experiment the positive_ids always has one paper.

You can create you own dataset with the same structure, and then train the citation recommendation system.

## Option 2: Download Processed Dataset
You can also download our **processed dataset** (and **pretrained models**) from [Google Drive](https://drive.google.com/drive/folders/1QwQuJsBOGEESFTgl-7wWbqcig7vJNlQ2?usp=sharing)

(There can be some additional information in the processed dataset other than what have been displayed in the examples above. They are irrelevant information.)

## Prefetching Part
In the following experiment, we use the "custom" dataset as an example. This dataset is the same as the ACL dataset. If you have created you dataset, you need to modify the config file at

MAIN/src.prefetch/config/YOUR_DATASET_NAME/global_config.cfg
### Training

This can take around 2h for this custom dataset (the same as ACL-200)


```python
!cd src/prefetch; python run.py -config_file_path config/custom/global_config.cfg -mode train
```

      3% 133/5000 [01:11<43:44,  1.85it/s]
    Traceback (most recent call last):
      File "train.py", line 193, in <module>
        loss = train_iteration(batch)
      File "train.py", line 36, in train_iteration
        return loss.item()
    KeyboardInterrupt
    Traceback (most recent call last):
      File "run.py", line 183, in <module>
        ))), shell = True
      File "/usr/lib/python3.7/subprocess.py", line 490, in run
        stdout, stderr = process.communicate(input, timeout=timeout)
      File "/usr/lib/python3.7/subprocess.py", line 956, in communicate
        self.wait()
      File "/usr/lib/python3.7/subprocess.py", line 1019, in wait
        return self._wait(timeout=timeout)
      File "/usr/lib/python3.7/subprocess.py", line 1653, in _wait
        (pid, sts) = self._try_wait(0)
      File "/usr/lib/python3.7/subprocess.py", line 1611, in _try_wait
        (pid, sts) = os.waitpid(self.pid, wait_flags)
    KeyboardInterrupt


This code will automatically handle the loop of  training -> updating paper embeddings -> updating prefetched candidates for contructing triplets -> resuming training.


In this case, the model checkpoint will be stored in the folder "MAIN/model/prefetch/custom/"; <br>
              the paper embeddings are stored in "MAIN/embedding/prefetch/custom/"; <br>
              the training log files are stored in "MAIN/log/prefetch/custom/"; 
              
The file MAIN/log/prefetch/custom/validate_NN.log contains the validation performance of each checkpoint during training. With this information, we can pick up the best-performance model for testing. 

You can specify where to store the checkpoint, log files and other parammeters by modifying the config/custom/global_config.cfg configuration file.

Note: **Before testing, after determining the best checkpoint, removing all the other checkpoints. If there are multiple checkpoints in MAIN/model/prefetch/custom/, the model will use the latest checkpoint by default.**

### Testing
To test the performance of the prefetching model, we need to first use the model checkpoint to compute the embedding for each paper in the database. This paper embedding is the index of the paper database, which is then used to perform nearest neighbor search. Then the next step is the test the prefetching performance.


Here I download the pretrained ACL model only for demonstration. If you are training the model using your custom data, you can wait until the training ends and select the best checkpoint based on the validation performance.


```python
try:
    os.system("rm -r model/prefetch/custom")
    os.makedirs("model/prefetch/custom")
except:
    pass
!cd model/prefetch/custom; gdown  https://drive.google.com/uc?id=13J5mtRg6t3Lcsn6fCdLJ1pZhtXnpREEq;
```

    Downloading...
    From: https://drive.google.com/uc?id=13J5mtRg6t3Lcsn6fCdLJ1pZhtXnpREEq
    To: /content/Local-Citation-Recommendation/model/prefetch/custom/model_batch_35000.pt
    100% 337M/337M [00:01<00:00, 199MB/s]


Step 1: Compute the embeddings of all papers in the database, using the trained text encoder (HAtten).


```python
!cd src/prefetch; python run.py -config_file_path config/custom/global_config.cfg -mode compute_paper_embedding
```

    100% 19776/19776 [00:43<00:00, 451.54it/s]


Step 2: Test the prefetching performance on the test set


```python
!cd src/prefetch; python run.py -config_file_path config/custom/global_config.cfg -mode test
```

    embedding loaded
    100% 9585/9585 [00:47<00:00, 200.26it/s]
    100% 8/8 [00:01<00:00,  4.48it/s]
    {'ckpt_name': '../../model/prefetch/custom/model_batch_35000.pt', 'recall': {10: 0.2811684924360981, 20: 0.3732916014606155, 50: 0.5021387584767867, 100: 0.6032342201356286, 200: 0.7003651538862806, 500: 0.8028169014084507, 1000: 0.8700052164840897, 2000: 0.9236306729264476}}
    Finished!


## Reranking Part

### Training
In order to train the reranker, we need to first create the training dataset. More specifically, for each query in the training set, we first use the trained HAtten prefetcher to prefetch a list of (2000) candidates. Then within the 2000 prefetched candidates we can construct triplets to train the reranker.


At this stage, we should have trained the prefetching model. We need to 1) compute paper embeddings for prefetching; 2) use the prefetching model to obtain prefetched candidates for each training example; 3) use the training examples with prefetched candidates to fine-tune SciBERT reranker. 


```python
!cd src/prefetch; python run.py -config_file_path config/custom/global_config.cfg -mode compute_paper_embedding
```

    100% 19776/19776 [00:43<00:00, 451.58it/s]



```python
!cd src/prefetch; python run.py -config_file_path config/custom/global_config.cfg -mode get_training_examples_with_prefetched_ids_for_reranking
```

    embedding loaded
    100% 10000/10000 [00:49<00:00, 200.75it/s]



```python
!cd src/prefetch; python run.py -config_file_path config/custom/global_config.cfg -mode get_val_examples_with_prefetched_ids_for_reranking
```

    embedding loaded
    100% 1000/1000 [00:05<00:00, 185.86it/s]


After get the prefetched ids for training and validstion set, we can start training the reranker: (This can take 2.5 h to finish one epoch on this custom dataset)


```python
!cd src/rerank; python train.py -config_file_path  config/custom/scibert/training_NN_prefetch.config
```

    Downloading: 100% 385/385 [00:00<00:00, 367kB/s]
    Downloading: 100% 228k/228k [00:00<00:00, 245kB/s]
    Downloading: 100% 442M/442M [00:06<00:00, 70.1MB/s]
    Some weights of the model checkpoint at allenai/scibert_scivocab_uncased were not used when initializing BertModel: ['cls.predictions.transform.dense.weight', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    /usr/local/lib/python3.7/dist-packages/transformers/optimization.py:310: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      FutureWarning,
      0% 0/10000 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1960: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
      2% 216/10000 [03:07<2:21:32,  1.15it/s]
    Traceback (most recent call last):
      File "train.py", line 161, in <module>
        loss = train_iteration( batch )
      File "train.py", line 37, in train_iteration
        loss.backward()
      File "/usr/local/lib/python3.7/dist-packages/torch/_tensor.py", line 396, in backward
        torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
      File "/usr/local/lib/python3.7/dist-packages/torch/autograd/__init__.py", line 175, in backward
        allow_unreachable=True, accumulate_grad=True)  # Calls into the C++ engine to run the backward pass
    KeyboardInterrupt
    ^C


## Use The HAtten Prefetcher and the SciBERT Reranker in Python Code

Before run this code, we should have trained the prefetching and the reranking models and know the path to the checkpoint of the saved model. (Here I download the pretrained model on ACL-200 for demonstration. If you train using your custom data, skip this and put the trained models' checkpoint to the corresponding model folder.)


```python
try:
    os.system("rm -r model/prefetch/custom")
    os.makedirs("model/prefetch/custom")
except:
    pass
!cd model/prefetch/custom; gdown  https://drive.google.com/uc?id=13J5mtRg6t3Lcsn6fCdLJ1pZhtXnpREEq;

try:
    os.system("rm -r model/rerank/custom")
    os.makedirs("model/rerank/custom/scibert/NN_prefetch")
except:
    pass
!cd model/rerank/custom/scibert/NN_prefetch; gdown  https://drive.google.com/uc?id=1DmSw6HR2W4fbUKp24K1_TREPhlLiQuEb;
```

    Downloading...
    From: https://drive.google.com/uc?id=13J5mtRg6t3Lcsn6fCdLJ1pZhtXnpREEq
    To: /content/Local-Citation-Recommendation/model/prefetch/custom/model_batch_35000.pt
    100% 337M/337M [00:01<00:00, 222MB/s]
    Downloading...
    From: https://drive.google.com/uc?id=1DmSw6HR2W4fbUKp24K1_TREPhlLiQuEb
    To: /content/Local-Citation-Recommendation/model/rerank/custom/scibert/NN_prefetch/model_batch_91170.pt
    100% 1.32G/1.32G [00:28<00:00, 46.1MB/s]


### Load models


```python
from citation_recommender import * 
prefetcher = Prefetcher( 
       model_path="model/prefetch/custom/model_batch_35000.pt",
       embedding_path="embedding/prefetch/custom/paper_embedding.pkl", ## make sure the papers embeddings have been computed
       gpu_list= [0,] 
)
reranker = Reranker( model_path = "model/rerank/custom/scibert/NN_prefetch/model_batch_91170.pt", 
                     gpu_list = [0,] )
```

    embedding loaded


    Some weights of the model checkpoint at allenai/scibert_scivocab_uncased were not used when initializing BertModel: ['cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.weight', 'cls.predictions.decoder.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


### Get paper recommendations

Then we can construct a query, and use the query to find n most relevant papers. The query is a dictionary containing 3 keys: "citing_title","citing_abstract" and "local_context". We can get some real example from the test set, or we can contruct a simple query as follows:

(### You can specify any other values, e.g., 100, 1000 or 2000. Note that the reranking time is proportional to the number of candidates to rerank.) 


```python
idx = 100
context_info = contexts[test_set[idx]["context_id"]]
citing_id = context_info["citing_id"]
refid = context_info["refid"]  ## The ground-truth cited paper

local_context = context_info["masked_text"]
citing_paper = papers[citing_id]
citing_title = citing_paper["title"]
citing_abstract = citing_paper["abstract"]
```


```python
citing_title, citing_abstract, local_context, refid
```




    ('Multiple System Combination for Transliteration',
     'We report the results of our experiments in the context of the NEWS 2015 Shared Task on Transliteration. We focus on methods of combining multiple base systems, and leveraging transliterations from multiple languages. We show error reductions over the best base system of up to 10% when using supplemental transliterations, and up to 20% when using system combination. We also discuss the quality of the shared task datasets.',
     ' score of the last prediction in the list. Our development experiments indicated that this method of combination was more accurate than a simpler method that uses only the prediction ranks. 4.2 RERANK TARGETCIT propose a reranking approach to transliteration to leverage supplemental representations, such as phonetic transcriptions and transliterations from other languages. The reranker utilizes many features',
     'N12-1044')




```python
candi_list = prefetcher.get_top_n(
  {
      "citing_title":citing_title,
      "citing_abstract":citing_abstract,
      "local_context":local_context
  }, 500
)
print(candi_list[:10])

for pos, cadi_id in enumerate(candi_list):
    if cadi_id == refid:
        print("The truely cited paper's id %s appears in position: %d among the prefetched ids."%( refid, pos ))
        break
if refid not in candi_list:
    print("The truely cited paper's id %s is not included in the prefetched ids"%( refid ))
```

    ['W15-3911', 'W09-3506', 'W10-2406', 'P04-1021', 'W10-2401', 'W12-4402', 'W09-3519', 'W09-3510', 'W11-3201', 'W12-4407']
    The truely cited paper's id N12-1044 appears in position: 49 among the prefetched ids.


Then we can rerank the prefetched candidates


```python
candidate_list = [  {"paper_id": pid,
                     "title":papers[pid].get("title",""),
                     "abstract":papers[pid].get("abstract","")}
                            for pid in candi_list ] 
## start reranking
reranked_candidate_list = reranker.rerank( citing_title,citing_abstract,local_context, candidate_list )
reranked_candidate_ids = [item["paper_id"] for item in reranked_candidate_list]
```


```python
for pos, cadi_id in enumerate(reranked_candidate_ids):
    if cadi_id == refid:
        print("The truely cited paper's id %s appears in position: %d among the reranked ids."%( refid, pos ))
        break
if refid not in reranked_candidate_ids:
    print("The truely cited paper's id %s is not included in the reranked ids"%( refid ))
```

    The truely cited paper's id N12-1044 appears in position: 2 among the reranked ids.


### Evaluation of the whole prefetching and reranking pipeline


We use HAtten to prefetch 100 candidates and then rerank them and we record the Recall@10 in the final recommendations  (We test this on 100 test examples only for demonstration)


```python
hit_list = []
top_K = 10

for idx in tqdm(range(100)):

    context_info = contexts[test_set[idx]["context_id"]]
    citing_id = context_info["citing_id"]
    refid = context_info["refid"]  ## The ground-truth cited paper

    local_context = context_info["masked_text"]
    citing_paper = papers[citing_id]
    citing_title = citing_paper["title"]
    citing_abstract = citing_paper["abstract"]

    candi_list = prefetcher.get_top_n(
        {
            "citing_title":citing_title,
            "citing_abstract":citing_abstract,
            "local_context":local_context
        }, 100  ## 100 candidates 
    )

    candidate_list = [  {"paper_id": pid,
                     "title":papers[pid].get("title",""),
                     "abstract":papers[pid].get("abstract","")}
                            for pid in candi_list ] 
    # start reranking
    reranked_candidate_list = reranker.rerank( citing_title,citing_abstract,local_context, candidate_list )
    reranked_candidate_ids = [item["paper_id"] for item in reranked_candidate_list]

    hit_list.append( refid in reranked_candidate_ids[:top_K])
```

      0%|          | 0/100 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1960: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    100%|██████████| 100/100 [05:13<00:00,  3.14s/it]

    The average recall@10:0.4500


    



```python
print("The average recall@%d: %.4f"%( top_K, np.mean(hit_list)))
```

    The average recall@10: 0.4500


This value is close to the results on ACL-200 in Table 4 in the paper, where we tested using full test set.

## References
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
