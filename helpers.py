import boto3
import warnings
import io
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold
from sklearn.linear_model import LogisticRegression
import random
from multiprocessing.dummy import Pool as ThreadPool
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from datetime import datetime

# from config import config

# BUCKET = config['bucket']
# PATH = config['project_path']
# RANDOM_STATE = config['random_state']

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')


def get_logger():
    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logFormatter)
    console_handler.setLevel(logging.NOTSET)
    logger.addHandler(console_handler)
    return logger


logger = get_logger()


def file_exists(path, bucket):
    obj_status = s3_client.list_objects(Bucket=bucket, Prefix=path)
    if obj_status.get('Contents'):
        return True
    else:
        return False


def check_data(*args):
    for a in args:
        if a.ndim == 1:
            logger.info(f'{a.shape} {np.unique(a, return_counts=True)}')
        else:
            logger.info(f'{a.shape}')


def get_s3(key, bucket):
    obj = s3.Object(bucket, key)
    return io.BytesIO(obj.get()['Body'].read())


def to_csv_s3(df, key, bucket, index=False):
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def np_save_s3(x, key, bucket):
    buf = io.BytesIO()
    np.save(buf, x)
    s3.Object(bucket, '{}.npy'.format(key)).put(Body=buf.getvalue())


def sp_save_s3(x, key, bucket):
    buf = io.BytesIO()
    sp.save_npz(buf, x)
    s3.Object(bucket, '{}.npz'.format(key)).put(Body=buf.getvalue())


def sample_data(x, y, pos_size, neg_size, seed):
    np.random.seed(seed)
    pos_samp_idx = np.random.choice(np.argwhere(y == 1).ravel(), pos_size, replace=False)
    np.random.seed(seed)
    neg_samp_idx = np.random.choice(np.argwhere(y == 0).ravel(), neg_size, replace=False)

    samp_idx = np.concatenate([pos_samp_idx, neg_samp_idx])
    x_samp = x[samp_idx, :]
    y_samp = y[samp_idx]

    return x_samp, y_samp


def threshold_metrics(actual, predicted, rank_best='gmean'):
    fpr, tpr, thresholds = roc_curve(actual, predicted)
    roc_auc = roc_auc_score(actual, predicted)

    p = sum(np.array(actual) == 1)
    n = sum(np.array(actual) == 0)
    perf_df = pd.DataFrame({'threshold': thresholds,
                            'tpr': tpr,
                            'tnr': 1 - fpr,
                            'b_acc': (tpr + (1 - fpr)) / 2.0,
                            'gmean': np.sqrt(tpr * (1 - fpr)),
                            'auc': [roc_auc] * len(fpr),
                            'acc': ((tpr * p) + ((1 - fpr) * n)) / (p + n)
                            })

    # get best threshold
    best_thresh = perf_df.sort_values(rank_best, ascending=False).iloc[0].to_dict()
    return best_thresh


class RatioRandomUnderSampler(RandomUnderSampler):
    def __init__(self, pos_ratio, random_state=0):
        self.pos_ratio = pos_ratio
        self.ratio_sampler = None
        super(RatioRandomUnderSampler, self).__init__(random_state=random_state)

    def fit(self, X, y):
        pos = len(y[y == 1])
        neg = int(pos * ((1 - self.pos_ratio) / self.pos_ratio))
        self.ratio_sampler = RandomUnderSampler(random_state=self.random_state, ratio={0: neg, 1: pos})
        self.ratio_sampler.fit(X, y)
        return self

    def sample(self, X, y):
        return self.ratio_sampler.sample(X, y)
