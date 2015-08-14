"""__author__ = 'Yicong Chen'

This is the library for Kaggle EEG signal detection, AAIO team.
"""

# Functions to load data with filename, subj=1~12, Data series=1~10, Event series=1~8
def get_filename(subj, series, type='data'):
    if subj in range(1,13) and series in range(1,9) and type in ['data', 'events']:
        FNAME = r"../30 Data/{0}/subj{1}_series{2}_{3}.csv"
        if type=='data':
            if series<9:
                FNAME.format('train',subj, series, type)
            else:
                FNAME.format('test', subj, series, type)
        else: #type='event'
            FNAME.format('test', subj, series, type)
        return FNAME
    else:
        print 'invalid subject number and series number.'
        return