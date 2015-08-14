"""__author__ = '310138649'
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.preprocessing import StandardScaler

# Functions to load data with filename, subj=1~12, train series=1~8, test series=9,10


