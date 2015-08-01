# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:45:46 2015

@author: yicong
"""

print (__doc__)

import sys
import numpy as np
import pandas as pd

from glob import glob
import matplotlib.pyplot as plt

t_lim = 4000 # upper time limit

try:
    subject = sys.argv[1]
    serie = sys.argv[2]
except IndexError:
    subject = 1
    serie = 1

#fnames =  glob("../30 Data/train/subj%d_series*_data.csv" % (subject))
events_fname = "../30 Data/train/subj%d_series%d_events.csv" % (subject, serie)
data_fname = "../30 Data/train/subj%d_series%d_data.csv" % (subject, serie)
events = pd.read_csv(events_fname)
data = pd.read_csv(data_fname)

events_names = list(events.columns[1:])
#events_data = np.array(events[events_names]).T

ch_names = list(data.columns[1:])
ch_data = np.array(data[ch_names]).T

for event_name in events_names[:1]:
    plt.figure()
    event_data = np.array(events[event_name])
    plt.plot(event_data[:t_lim])
    plt.title(event_name)
    for i, ch_name in enumerate(ch_names[2:8]):
        ch_data = np.array(data[ch_names])
        plt.figure()
        plt.plot(ch_data[:t_lim])
        plt.title(ch_name)

plt.show()