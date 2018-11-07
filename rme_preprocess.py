import sys
import datetime
import json
import os
import time

import numpy as np


import pandas as pd
import scipy.sparse
import utils


t1 = time.time()

DEBUG_MODE = False
REVERSE = False # if set to true, project with larger timestamp will have smaller id

def timestamp_to_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
def date_to_timestamp(date):
    return utils.convert_to_datetime(date)
def get_count(df, id):
    count_groupbyid = df[[id]].groupby(id, as_index=False)
    count = count_groupbyid.size()
    return count
#remove users who backed less than min_pc projects, and projects with less than min_uc users:
def filter(df, min_pc=5, min_uc=5):
    #keep users who backed at least min_pc projects
    current_size = 0
    next_size = df.shape[0]
    iter = 1
    while(current_size != next_size):
        print 'filter with loop %d, size: %d'%(iter, df.shape[0])
        iter += 1
        current_size = df.shape[0]
        if min_pc > 0:
            usercount = get_count(df, 'userId')
            df = df[df['userId'].isin(usercount.index[usercount >= min_pc])]
        if current_size != next_size:
            continue
        # keep projects which are backed by at least min_uc users
        # After doing this, some of the projects will have less than min_uc users, if we remove them,
        # some of users may have less than min_pc backed projects
        # ==> keep looping until stable.
        if min_uc > 0:
            projectcount = get_count(df, 'movieId')
            df = df[df['movieId'].isin(projectcount.index[projectcount >= min_uc])]
        next_size = df.shape[0]
        # Update both usercount and songcount after filtering
    usercount, projectcount = get_count(df, 'userId'), get_count(df, 'movieId')
    return df, usercount, projectcount


#read project-timestamp into panda dataframe and sort by timestamp.
PROJECT_INFO_PATH = "data/ml/ratings.csv"
user_pro_data = pd.read_csv(PROJECT_INFO_PATH, header=0,  sep=',') #userId,movieId,rating,timestamp
# all_data = user_pro_data
user_pro_data = user_pro_data.drop_duplicates(['userId','movieId'])
print 'project-info-path: %d'% user_pro_data.shape[0]
user_pro_data = user_pro_data[user_pro_data['rating'] > 3.5]
start_t = time.mktime(datetime.datetime.strptime("1995-01-01", "%Y-%m-%d").timetuple())
user_pro_data = user_pro_data[user_pro_data['timestamp'] > start_t]




print 'After removing projects with empty ts of project-info-path and rating >= 4: %d', user_pro_data.shape[0]
if REVERSE:
    user_pro_data = user_pro_data.sort_index(by=['timestamp'], ascending=False) #smaller id with larger ts
else:
    user_pro_data = user_pro_data.sort_index(by=['timestamp'], ascending=True) #smaller id with smaller ts
#print project_info_data
tstamp = np.array(user_pro_data['timestamp'])
print("Time span of the dataset: From %s to %s" %
      (timestamp_to_date(np.min(tstamp)), timestamp_to_date(np.max(tstamp))))
# apparently the timestamps are ordered, check to make sure
for i in xrange(tstamp.size - 1):
    if tstamp[i] < tstamp[i + 1] and REVERSE:
        print 'must reorder'
        sys.exit(1)
    if tstamp[i] > tstamp[i + 1] and not REVERSE:
        print 'must reorder'
        sys.exit(1)
print user_pro_data
tr_vd_raw_data = user_pro_data[:int(0.8 * user_pro_data.shape[0])]
split_time = tstamp[int(0.8 * user_pro_data.shape[0])]
tr_vd_raw_data, user_activity, item_popularity = filter(tr_vd_raw_data)
sparsity = 1. * tr_vd_raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

DATA_DIR = 'data/rec_data/'
CONTAINED_DIR = "all"


unique_uid = user_activity.index
unique_sid = item_popularity.index
song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))


def numerize(tp):
    uid = map(lambda x: user2id[x], tp['userId'])
    sid = map(lambda x: song2id[x], tp['movieId'])
    tp['userId'] = uid
    tp['movieId'] = sid
    return tp[['timestamp', 'userId', 'movieId']]

test_raw_data = user_pro_data[int(0.8 * len(user_pro_data)):]
test_raw_data = test_raw_data[test_raw_data['movieId'].isin(unique_sid)]
test_raw_data = test_raw_data[test_raw_data['userId'].isin(unique_uid)]
test_data = numerize(test_raw_data)
for FOLD, seed_init in enumerate([13579,98765,24680,14689,97531]):
    np.random.seed(seed_init)
    n_ratings = tr_vd_raw_data.shape[0]
    vad = np.random.choice(n_ratings, size=int(0.125 * n_ratings), replace=False)
    vad_idx = np.zeros(n_ratings, dtype=bool)
    vad_idx[vad] = True

    vad_raw_data = tr_vd_raw_data[vad_idx]
    train_raw_data = tr_vd_raw_data[~vad_idx]
    print "There are total of %d unique users in the training set and %d unique users in the entire dataset" % \
    (len(pd.unique(train_raw_data['userId'])), len(unique_uid))
    print "There are total of %d unique items in the training set and %d unique items in the entire dataset" % \
    (len(pd.unique(train_raw_data['movieId'])), len(unique_sid))
    train_sid = set(pd.unique(train_raw_data['movieId']))
    left_sid = list()
    for i, sid in enumerate(unique_sid):
        if sid not in train_sid:
            left_sid.append(sid)
    move_idx = vad_raw_data['movieId'].isin(left_sid)
    train_raw_data = train_raw_data.append(vad_raw_data[move_idx])
    vad_raw_data = vad_raw_data[~move_idx]
    print "There are total of %d unique items in the training set and %d unique items in the entire dataset" % \
    (len(pd.unique(train_raw_data['movieId'])), len(unique_sid))

    print len(train_raw_data), len(vad_raw_data), len(test_raw_data)
    train_timestamp = np.asarray(tr_vd_raw_data['timestamp'])
    print("train: from %s to %s" % (timestamp_to_date(train_timestamp[0]),
                                    timestamp_to_date(train_timestamp[-1])))

    test_timestamp = np.asarray(test_raw_data['timestamp'])
    print("test: from %s to %s" % (timestamp_to_date(test_timestamp[0]),
                                   timestamp_to_date(test_timestamp[-1])))


    train_data = numerize(train_raw_data)
    unique_uid = sorted(train_data['userId'].unique())
    unique_sid = sorted(train_data['movieId'].unique())
    with open(os.path.join('data/rec_data/all/', 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)
    with open(os.path.join('data/rec_data/all/', 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    train_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'train_fold%d.csv'%FOLD), index=False)
    vad_data = numerize(vad_raw_data)
    vad_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'validation_fold%d.csv'%FOLD), index=False)

    test_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'test_fold%d.csv'%FOLD), index=False)



    t2 = time.time()
    print 'Time : %d seconds'%(t2 - t1)


    #save the negative-item matrix
    tp = pd.read_csv(PROJECT_INFO_PATH, header=0,  sep=',') #userId,movieId,rating,timestamp
    tp = tp[tp['timestamp'] < split_time]
    tp = tp[tp['rating'] <= 2]
    tp = tp[(tp['userId'].isin(user2id.keys())) & (tp['movieId'].isin(song2id.keys()))]
    neg_data = numerize(tp)
    neg_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'train_neg_fold%d.csv'%FOLD), index=False)
    # print neg_data
