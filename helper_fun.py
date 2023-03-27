import pickle
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



def date_range(start_date, end_date):
    """function is utilized to generate date range data of munutes level granularity which is helpful for current assignment
    
    arguments:
    start_date(datetime): start point of the daterange
    end_date(datetime): endpoint of the daterange
    
    returns:
    
    results(list): list of datarange
    
    """
    
    result = []
    nxt = start_date
    delta =  timedelta(minutes=1)
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return result



def fill_data(my_scr):
    """ A data filler which fills gaps with previous data values
    
    """
    
    full_stack = []
    filler_stack=[]
    counter =0 
    for i in my_scr:
        if not np.isnan(i):
            full_stack.extend(filler_stack)
            full_stack.append(i)
            filler_stack = []
            counter=0
        else:
            if len(full_stack)>counter:
                filler_stack.append(full_stack[-counter])
            else:
                filler_stack.append(np.NaN)
            counter+=1

    return full_stack


def stam_outlier(scr):
    
    """ function to stam the outliers in the dataset"""
    full_stack = []
    s3_limit = 6*np.std(scr)
    for i in scr:
        if i < s3_limit :
            full_stack.append(i)
        else:
            full_stack.append(s3_limit)
    return full_stack



# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = [],[]
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)