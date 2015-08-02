# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:17:14 2015

@author: yicong

another fork after announcement
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
 
#############function to read data###########
FNAME = "../30 Data/{0}/subj{1}_series{2}_{3}.csv"
def load_data(subj, series=range(1,9), prefix = 'train'):
    data = [pd.read_csv(FNAME.format('train',subject,s,'data'), index_col=0) for s in series]
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
    for fc in np.linspace(0,1,11)[1:]:
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


#%%########### Initialize ####################################################
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

subjects = range(1,13)
idx_tot = []
scores_tot = []

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects[:1]:

    X_train, y = load_data(subject, range(1,7))
#    X_test, idx = load_data(subject,[9,10],'test')
    X_test, idx = load_data(subject,[7,8], 'test')

################ Train classifiers ###########################################
    #lda = LDA()
    lr = LogisticRegression()
    
    X_train, scaler = compute_features(X_train)
    X_test = compute_features(X_test, scaler)   #pass the learned mean and std to normalized test data
    
    y = np.concatenate(y,axis=0)
    scores = np.empty((X_test.shape[0],6))
    
    downsample = 50
    for i in range(6):
        print('Train subject %d, class %s' % (subject, cols[i]))
        lr.fit(X_train[::downsample,:], y[::downsample,i])
        #lda.fit(X_train[::downsample,:], y[::downsample,i])
       
        scores[:,i] = 1*lr.predict_proba(X_test)[:,1]# + 0.5*lda.predict_proba(X_test)[:,1]

    scores_tot.append(scores)
    idx_tot.append(np.concatenate(idx))
    
#%%########### submission file ################################################
submission_file = 'Submission.csv'
# create pandas object for submission
submission = pd.DataFrame(index=np.concatenate(idx_tot),
                          columns=cols,
                          data=np.concatenate(scores_tot))

# write file
#submission.to_csv(submission_file,index_label='id',float_format='%.3f')

y7 = pd.read_csv(r'../30 Data/train/subj1_series7_events.csv', index_col=0)
y7_events = y7.values
y8 = pd.read_csv(r'../30 Data/train/subj1_series8_events.csv', index_col=0)
y8_events = y8.values
y = np.concatenate([y7_events, y8_events], axis=0)
print roc_auc_score(y, submission.values)