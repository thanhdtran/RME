# RME

This repo contains source code for our paper: "Regularizing Matrix Factorization with User and Item Embeddings for Recommendation" published in CIKM 2018

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
**_produce_negative_cooccurrence.py --dataset ml10m_**


# RUNNING:
## Step 1: produce user-user co-occurrence matrix and item-item co-occurrence matrix
python produce_positive_cooccurrence.py --dataset ml10m

## Step 2: run RME with our user-oriented EM-like algorithm to infer disliked items for users:
python rme_rec.py --dataset ml10m --model rme --neg_item_inference 1 --lam 1.0 --lam_emb 1.0

where:
- lam: is the regularization hyper-parameter for user and item latent factors (alpha and beta)
- lam_emb: is the regularization hyper-parameter for user and item context latent factors (gamma, theta, delta)

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



