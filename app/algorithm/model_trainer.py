#!/usr/bin/env python

import os, warnings, sys 
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split


import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils

#import algorithm.scoring as scoring
from algorithm.model.recommender import Recommender, get_data_based_model_params
from algorithm.utils import get_model_config


# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
    
    
    # split train data into train and validation data     
    train_data, valid_data = utils.get_train_valid_split(data, model_cfg["valid_split"])
    # print('After train/valid split, train_data shape:',  train_data.shape, 'valid_data shape:',  valid_data.shape)
        
    
    # preprocess data
    print("Pre-processing data...")
    train_data_tup, valid_data_tup, preprocess_pipe = preprocess_data(train_data, valid_data, data_schema)  
              
    # Create and train model     
    print('Fitting model ...')  
    model, history = train_model(train_data_tup, valid_data_tup, hyper_params, verbose=1)    
    
    return preprocess_pipe, model, history


def train_model(train_data_tup, valid_data_tup, hyper_params, verbose=0):   
    # get model hyper-parameters  that are data-dependent (in this case N = num_users, and M = num_items)   
    train_X_R =  train_data_tup[0]
    data_based_params = get_data_based_model_params(train_X_R) 
    
    model_params = { **data_based_params, **hyper_params }
    
    # Create and train model   
    model = Recommender(  **model_params )  
    # model.summary()  
    
    # fit model
    history = model.fit( 
            train_data_tup=train_data_tup, 
            valid_data_tup=valid_data_tup, 
            epochs=100,
            verbose=verbose,
        )  
    
    return model, history


def preprocess_data(train_data, valid_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(data_schema)   
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_X_R, train_X_M, train_Y_R, train_Y_M, train_user_ids_int = preprocess_pipe.fit_transform(train_data)
    train_data_tup = train_X_R, train_X_M, train_Y_R, train_Y_M
    # print("Processed train data shapes", train_X_R.shape, train_X_M.shape, train_Y_R.shape, train_Y_M.shape)
      
    if valid_data is not None:
        valid_X_R, valid_X_M, valid_Y_R, valid_Y_M, valid_user_ids_int = preprocess_pipe.transform(valid_data)
        valid_data_tup = valid_X_R, valid_X_M, valid_Y_R, valid_Y_M
        # print("Processed valid data shapes", valid_X_R.shape, valid_X_M.shape, valid_Y_R.shape, valid_Y_M.shape)
    else: 
        valid_data_tup = None
    return train_data_tup, valid_data_tup, preprocess_pipe 


