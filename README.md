# RME
data format: userId,movieId

data for running our source code: ml10m.
We preprocessed it and splitted into train, vad/dev, test. Their paths are:

data/ml10m/train.csv

data/ml10m/test.csv

data/ml10m/validation.csv

format of user and disliked items: userId,movieId

if we have information about users and disliked items, saved it to data/ml10m/train_neg.csv


RUNNING:
#produce user-user co-occurrence matrix and item-item co-occurrence matrix
python produce_positive_cooccurrence.py --dataset ml10m

next, we run RME with our user-oriented EM-like algorithm as:
python rme_rec.py --dataset ml10m --model rme --neg_item_inference 1 --lam 1.0 --lam_emb 1.0

where lam: is the regularization hyper-parameter for user and item latent factors (alpha and beta)
lam_emb: is the regularization hyper-parameter for user and item context latent factors (gamma, theta, delta)




This repo contains source code for our paper:
"Regularizing Matrix Factorization with User and Item Embeddings for Recommendation"
published in CIKM 2018

