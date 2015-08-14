"""__author__ = '310138649'
"""

import aaio
import itertools

from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.preprocessing import StandardScaler

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

X = aaio.read_file(aaio.get_filename(1,1,'data'))