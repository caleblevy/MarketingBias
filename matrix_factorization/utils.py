import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import dataset
import os
import heapq
import statsmodels.api as sm
from statsmodels.formula.api import ols

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)

MODEL_DIR = os.path.join(BASE_DIR, "model/")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


class Model(object):
    def __init__(self, METHOD_NAME, config):
        self.config = config
        config_string = "_".join([str(config[k]) for k in sorted(list(config.keys())) if config[k] is not None])

        self.MODEL_NAME = "_".join([METHOD_NAME, config_string])

    def assign_data(self, n_user, n_item,
                    user_attr, item_attr,
                    user_attr_ids, item_attr_ids,
                    data_train, data_validation,
                    N_MAX=500000):
        self.n_user = n_user
        self.n_item = n_item
        self.item_attr = item_attr
        self.user_attr = user_attr
        self.item_attr_ids = item_attr_ids
        self.user_attr_ids = user_attr_ids
        self.item_user_attr_map = np.arange(len(item_attr_ids)*len(user_attr_ids)).reshape(
            len(item_attr_ids), len(user_attr_ids))
        self.data_train = data_train
        self.data_validation = data_validation
        self.N_MAX = min(data_train.shape[0], N_MAX)

    def assign_neg_samples(self, neg_samples):
        self.neg_samples = neg_samples
        self.N_NEG = neg_samples.shape[1]

    def next_train_batch(self, BATCH_SIZE):
        pass

    def get_validation_batches(self, dataInput, BATCH_SIZE):
        pass

    def model_constructor(self):
        pass

    def train(self):
        BATCH_SIZE = self.config['batch_size']
        MODEL_NAME = self.MODEL_NAME

        EPOCHS = 1000
        max_noprogress = 10

        batches_validation = self.get_validation_batches(
            self.data_validation, BATCH_SIZE)

        config = tf.ConfigProto()
        with tf.Graph().as_default(), tf.Session(config=config) as session:

            variables, scores, losses, errors, optimizers = self.model_constructor()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            _loss_train_min = 1e10
            _loss_vali_min = 1e10
            _loss_vali_old = 1e10
            n_noprogress = 0

            for epoch in range(1, EPOCHS):
                _count, _count_sample = 0, 0
                _loss_train = [0 for _ in range(len(losses))]

                for _vars in self.next_train_batch(BATCH_SIZE):
                    feed = dict(zip(variables, _vars))

                    _loss_batch, _ = session.run([losses, optimizers],
                                                 feed_dict=feed)
                    for _i, _l in enumerate(_loss_batch):
                        _loss_train[_i] += _l*_vars[0].shape[0]

                    _count += 1.0
                    _count_sample += _vars[0].shape[0]

                for _i in range(len(_loss_train)):
                    _loss_train[_i] /= _count_sample

                if _loss_train[0] < _loss_train_min:
                    _loss_train_min = _loss_train[0]

                _count, _count_sample = 0, 0
                _loss_vali = [0 for _ in range(len(losses))]
                for _vars in batches_validation:
                    feed = dict(zip(variables, _vars))

                    _loss_batch = session.run(losses, feed_dict=feed)
                    for _i, _l in enumerate(_loss_batch):
                        _loss_vali[_i] += _l*_vars[0].shape[0]
                    _count += 1
                    _count_sample += _vars[0].shape[0]

                for _i in range(len(_loss_vali)):
                    _loss_vali[_i] /= _count_sample

                if _loss_vali[0] <= _loss_vali_min:
                    _loss_vali_min = _loss_vali[0]
                    n_noprogress = 0
                    saver.save(session, os.path.join(
                        MODEL_DIR, self.MODEL_NAME + ".model.ckpt"))

                if (_loss_vali[0] > _loss_vali_old) or (_loss_train[_i] > _loss_train_min):
                    n_noprogress += 1
                _loss_vali_old = _loss_vali[0]

                if n_noprogress > max_noprogress:
                    break
            saver.restore(session, os.path.join(
                MODEL_DIR, self.MODEL_NAME + ".model.ckpt"))

    def batch(self, dataInput, BATCH_SIZE):
        res = []
        for i in range(0, dataInput.shape[0], BATCH_SIZE):
            res.append(dataInput[i:(i+BATCH_SIZE), :])
        return res

    # make sure the first 3 vars are user, item, rating
    def get_rating(self, df_validation, df_test):
        BATCH_SIZE = self.config['batch_size']

        res = []
        with tf.Graph().as_default(), tf.Session() as session:
            variables, scores, losses, errors, optimizers = self.model_constructor()

            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(MODEL_DIR, self.MODEL_NAME + ".model.ckpt"))

            batches_vali = self.batch(df_validation[['user_id', 'item_id', 'rating']].values, BATCH_SIZE)

            _errors = np.array([])
            for _vars in batches_vali:
                feed = {variables[0]: _vars[:, 0].astype(np.int32),
                        variables[1]: _vars[:, 1].astype(np.int32),
                        variables[2]: _vars[:, 2].astype(np.float32)}
                _errors_batch = session.run(errors, feed_dict=feed)
                _errors = np.append(_errors, _errors_batch)

            df_validation['error'] = _errors

            batches_test = self.batch(df_test[['user_id', 'item_id', 'rating']].values, BATCH_SIZE)

            _errors = np.array([])

            for _vars in batches_test:
                feed = {variables[0]: _vars[:, 0].astype(np.int32),
                        variables[1]: _vars[:, 1].astype(np.int32),
                        variables[2]: _vars[:, 2].astype(np.float32)}

                _errors_batch = session.run(errors, feed_dict=feed)
                _errors = np.append(_errors, _errors_batch)

            true_rating = np.array(df_test['rating'])
            df_test['inferred_rating'] = true_rating + _errors

            print('mse', (_errors * _errors).mean())

        return df_test[['user_id', 'item_id', 'inferred_rating']]
