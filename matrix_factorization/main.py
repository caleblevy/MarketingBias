import pandas as pd
import os
import sys
import json

import argparse

import dataset
from model import MF

def mf_rating(data, dim=10, lbda=10, learning_rate=0.001, c=0.5, batch_size=512,
              protect_item_group=1, protect_user_group=1, protect_user_item_group=1):
    data = data.astype('float64')
    myData = dataset.Dataset(data)
    
    config = {'hidden_dim': dim,
              'lbda': lbda,
              'learning_rate': learning_rate,
              'batch_size': batch_size,
              'C': c,
              'protect_item_group': protect_item_group,
              'protect_user_group': protect_user_group,
              'protect_user_item_group': protect_user_item_group}
    myModel = MF(config)

    columns = ['user_id', 'item_id', 'rating']

    myModel.assign_data(myData.n_user, myData.n_item,
                        myData.user_attr, myData.item_attr,
                        myData.user_attr_ids, myData.item_attr_ids,
                        myData.data[columns].loc[myData.data['split'] == 0].values.astype(int),
                        myData.data[columns].loc[myData.data['split'] == 1].values.astype(int))

    #apply_user_item_map(ratings, myData.get_user_item_train_map())

    myModel.train()

    columns = ['user_id', 'item_id', 'rating', 'model_attr', 'user_attr']
    ratings = myModel.get_rating(myData.data[columns].loc[myData.data['split'] == 1],
                                 myData.data[columns].loc[myData.data['split'] == 2])

    ratings = apply_user_item_map(ratings, myData.user_ids, myData.item_ids)
    ratings[['user_id', 'item_id']] = ratings[['user_id', 'item_id']].astype('Int64')
    
    return ratings

def apply_user_item_map(data, user_map, item_map):
    mapped_users = data['user_id'].values
    mapped_items = data['item_id'].values

    original_users = [user_map[user] for user in mapped_users]
    original_items = [item_map[item] for item in mapped_items]

    data['user_id'] = original_users
    data['item_id'] = original_items

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="specify a dataset from [modcloth, electronics]")
    parser.add_argument('--protect_item_group', default=1, type=int)
    
    parser.add_argument('--protect_user_group', default=1, type=int)
    parser.add_argument('--protect_user_item_group', default=1, type=int)

    args = parser.parse_args()

    dataname = os.path.join(os.path.dirname(__file__), '..', 'datasets/modcloth/raw', 'df_' + args.dataset + '.csv')
    data = pd.read_csv(dataname)

    mf_rating(data=data,
              protect_item_group=args.protect_item_group,
              protect_user_group=args.protect_user_group,
              protect_user_item_group=args.protect_user_item_group)
