# RME

This repo contains source code for our paper: "Regularizing Matrix Factorization with User and Item Embeddings for Recommendation" published in CIKM 2018. We implemented using multi-threads, so it is very fast to run with big datasets.

# DATA FORMAT 


#### Data format:
- First line: the header "userId,movieId"
- Second line --> last line: [userId],[movieId]

### data for running our source code: ml10m.
We preprocessed it and splitted into train, vad/dev, test. Their paths are:

- data/ml10m/train.csv

- data/ml10m/test.csv

- data/ml10m/validation.csv

### format of user and disliked items: same as previous format: 
- First line: the header "userId,movieId"
- Second line --> last line: [userId],[movieId]

### When we have available users and dislike items:
do 2 steps:
- saved it to data/ml10m/train_neg.csv
- build the disliked item-item co-occurrence by running (assume that the dataset is ml10m):
**produce_negative_cooccurrence.py --dataset ml10m**


# RUNNING:
### Step 1: produce user-user co-occurrence matrix and item-item co-occurrence matrix
**python produce_positive_cooccurrence.py --dataset ml10m**

### Step 2: run RME with our user-oriented EM-like algorithm to infer disliked items for users:
**python rme_rec.py --dataset ml10m --model rme --neg_item_inference 1 --n_factors 40 --reg 1.0 --reg_embed 1.0**

where:
- <code>model</code>: the model to run. There are 3 choices: <code>rme</code> (our model), <code>wmf</code>, <code>cofactor</code>.
- <code>reg</code>: is the regularization hyper-parameter for user and item latent factors (alpha and beta).
- <code>reg_emb</code>: is the regularization hyper-parameter for user and item context latent factors (gamma, theta, delta).
- <code>n_factors</code>: number of latent factors (or embedding size). Default: n_factors = 40.
- <code>neg_item_inference</code>: whether or not running our user-oriented EM like algorithm for sampling disliked items for users. In case we have available user-disliked_items --> set this to 0.
- <code>neg_item_inference</code>: negative sample ratio per user. If a user consumed 10 items, and this neg_sample_ratio = 0.2 --> randomly sample 2 negative items for the user. Default: 0.2.

#### other hyper-parameters:
- <code>s</code>: the shifted constant, which is a hyper-parameter to control density of SPPMI matrix. Default: s = 1.
- <code>data_path</code>: path to the data. Default: data.
- <code>saved_model_path</code>: path to saved the optimal model using validation/development dataset. Default: MODELS.

You may get some results like:
```
top-5 results: recall@5 = 0.1559, ndcg@5 = 0.1613, map@5 = 0.1076
top-10 results: recall@10 = 0.1513, ndcg@10 = 0.1547, map@10 = 0.0851
top-20 results: recall@20 = 0.1477, ndcg@20 = 0.1473, map@20 = 0.0669
top-50 results: recall@50 = 0.1819, ndcg@50 = 0.1553, map@50 = 0.0562
top-100 results: recall@100 = 0.2533, ndcg@100 = 0.1825, map@100 = 0.0579
```

### running some baselines: Cofactor, WMF:
- Running cofactor:

**python rme_rec.py --dataset ml10m --model cofactor --n_factors 40 --reg 1.0 --reg_embed 1.0**

You may get the results like:
```
top-5 results: recall@5 = 0.1522, ndcg@5 = 0.1537, map@5 = 0.1000
top-10 results: recall@10 = 0.1383, ndcg@10 = 0.1425, map@10 = 0.0756
top-20 results: recall@20 = 0.1438, ndcg@20 = 0.1391, map@20 = 0.0606
top-50 results: recall@50 = 0.1762, ndcg@50 = 0.1484, map@50 = 0.0518
top-100 results: recall@100 = 0.2545, ndcg@100 = 0.1783, map@100 = 0.0540
```
- Running WMF:

**python rme_rec.py --dataset ml10m --model wmf --n_factors 40 --reg 1.0 --reg_embed 1.0**

You may get the results like:
```
top-5 results: recall@5 = 0.1258, ndcg@5 = 0.1283, map@5 = 0.0810
top-10 results: recall@10 = 0.1209, ndcg@10 = 0.1231, map@10 = 0.0624
top-20 results: recall@20 = 0.1290, ndcg@20 = 0.1230, map@20 = 0.0507
top-50 results: recall@50 = 0.1641, ndcg@50 = 0.1349, map@50 = 0.0442
top-100 results: recall@100 = 0.2375, ndcg@100 = 0.1640, map@100 = 0.0470
```


# CITATION:

If you use this Caser in your paper, please cite the paper:

```
@inproceedings{tran2018regularizing,
  title={Regularizing Matrix Factorization with User and Item Embeddings for Recommendation},
  author={Tran, Thanh and Lee, Kyumin and Liao, Yiming and Lee, Dongwon},
  booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  pages={687--696},
  year={2018},
  organization={ACM}
}
```



