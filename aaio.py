"""__author__ = 'Yicong Chen'

This is the library for Kaggle EEG signal detection, AAIO team.
"""

import pandas as pd
import numpy as np

from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.preprocessing import StandardScaler

# # Functions to load data with filename, subj=1~12, Data series=1~10, Event series=1~8
# def get_filename(subj, series, type='data'):
#     if subj in range(1,13) and series in range(1,9) and type in ['data', 'events']:
#         FNAME = r"../30 Data/{0}/subj{1}_series{2}_{3}.csv"
#         if type=='data':
#             if series<9:
#                 fn = FNAME.format('train', subj, series, type)
#             else:
#                 fn = FNAME.format('test', subj, series, type)
#         else: #type='event'
#             fn = FNAME.format('train', subj, series, type)
#         return fn
#     else:
#         print 'invalid subject number, series number or type[data,events].'
#         return
#
# def read_file(fn):
#     data = pd.read_csv(fn)
#     ch_names = data.columns[1:]
#     ch_data = data[ch_names]
#     ch_data = np.array(ch_data)
#     return ch_data

def load_data(subject, series=range(1,9), prefix = 'train'):
    FNAME = "../30 Data/{0}/subj{1}_series{2}_{3}.csv"
    data = [pd.read_csv(FNAME.format(prefix,subject,s,'data'), index_col=0) for s in series]
    idx = [d.index for d in data]
    data = [d.values.astype(float) for d in data]
    if prefix == 'train':
        events = [pd.read_csv(FNAME.format(prefix,subject,s,'events'), index_col=0).values for s in series]
        return data, events
    else:
        return data, idx

def compute_features(X, scale=None):
    X0 = [x[:,0] for x in X]
    X = np.concatenate(X, axis=0)
    F = [];
    fcrange = np.linspace(0,1,12)[1:]
    for fc in fcrange:
        b,a = butter(3,fc/250.0,btype='lowpass')
        F.append(np.concatenate([lfilter(b,a,x0) for x0 in X0], axis=0)[:,np.newaxis])
    F = np.concatenate(F, axis=1)
    F = np.concatenate((X,F,F**2), axis=1)

    if scale is None:
        scale = StandardScaler()
        F = scale.fit_transform(F)
        return F, scale
    else:
        F = scale.transform(F)
        return F