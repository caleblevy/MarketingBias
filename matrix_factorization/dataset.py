import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, data):
        data = data.copy()
        data['item_id'].loc[data['item_id'].isna()] = ''
        data['user_id'].loc[data['user_id'].isna()] = ''

        item_id_vals, item_ids = pd.factorize(data['item_id'].values)
        user_id_vals, user_ids = pd.factorize(data['user_id'].values)
        item_attr_vals, item_attr_ids = pd.factorize(data['model_attr'].values)
        user_attr_vals, user_attr_ids = pd.factorize(data['user_attr'].values)

        tmp = dict(zip(data['item_id'].values, item_attr_vals))
        self.item_attr = np.array([tmp[_i] for _i in item_ids], dtype=int)
        tmp = dict(zip(data['user_id'].values, user_attr_vals))
        self.user_attr = np.array([tmp[_i] for _i in user_ids], dtype=int)
        
        data['item_id'] = item_id_vals
        data['user_id'] = user_id_vals
    
        self.item_ids = item_ids
        self.user_ids = user_ids
        self.item_attr_ids = item_attr_ids
        self.user_attr_ids = user_attr_ids

        self.n_item = len(data['item_id'].unique())
        self.n_user = len(data['user_id'].unique())

        self.data = data[['user_id','item_id','rating','split','model_attr','user_attr']]

    def get_user_item_train_map(self):
        data = self.data
        
        user_item_train_map = (self.data.loc[(self.data['rating']>=4) & (self.data['split'] == 0)]).groupby(
            ['user_id'])['item_id'].apply(list).to_dict()
        
        return user_item_train_map
