import preprocessing as pp
import exploring as ex

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



def data_preparation(filepath, list_name, encode_cont = True, encode_disc = True,
                     scale = True, nan = True):
    
    start_data = pp.get_data(filepath)
    est, not_est = pp.split_est(start_data)
    
    if nan == True:
        nan_est, est = pp.split_NaN(est)
        nan_not_est, not_est = pp.split_NaN(not_est)
        
    if encode_cont == True & encode_disc == True:
        est = pp.encode(est)
        not_est = pp.encode(not_est)
        if scale == True:
            est = pp.scale_data(est)
            not_est = pp.scale_data(not_est)
        
    elif encode_cont == True & encode_disc == False:
        cont1, disc1 = pp.split_cont_disc(est)
        cont2, disc2 == pp.split_cont_disc(not_est)
        cont1 = pp.encode(cont1)
        cont2 = pp.encode(cont2)   
        if scale == True:
            cont1 = pp.scale_data(cont1)
            cont2 = pp.scale_data(cont2)
        est = pd.concat([cont1, disc1], axis = 1)
        not_est = pd.concat([cont2, disc2], axis = 1)
        est = pd.DataFrame(est)
        not_est = pd.DataFrame(not_est)
        print("discrete data cannot be scaled unless encoded first: use encode_disc = True")
    
    elif encode_cont == False & encode_disc == True:
        cont1, disc1 = pp.split_cont_disc(est)
        cont2, disc2 == pp.split_cont_disc(not_est)
        disc1 = pp.encode(disc1)
        disc2 = pp.encode(disc2)
        est = pd.concat([cont1, disc1], axis = 1)
        not_est = pd.concat([cont2, disc2], axis = 1)
        est = pd.DataFrame(est)
        not_est = pd.DataFrame(not_est)
        if scale == True:
            est = pp.scale_data(est)
            not_est = pp.scale_data(not_est)
    
    elif encode_cont == False & encode_disc == False:
        if scale == True:
            cont1, disc1 = pp.split_cont_disc(est)
            cont2, disc2 = pp.split_cont_disc(not_est)
            cont1 = pp.scale_data(cont1)
            cont2 = pp.scale_data(cont2)
            est = pd.concat([cont1, disc1], axis = 1)
            not_est = pd.concat([cont2, disc2], axis = 1)
            est = pd.DataFrame(est)
            not_est = pd.DataFrame(not_est)
            print("discrete data cannot be scaled unless encoded first: use encode_disc = True")
  
    est = est[list_name]
    not_est = not_est[list_name]
    return(est, not_est)

