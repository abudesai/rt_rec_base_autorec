import os, sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

import algorithm.preprocessing.preprocessors as preprocessors



PREPROCESSOR_FNAME = "preprocessor.save"


def get_preprocess_pipeline(pp_params, model_cfg): 
    
    cfg_pp_steps = model_cfg["pp_params"]["pp_step_names"]
    cfg_pp_int_fields = model_cfg["pp_params"]["int_fields"]    
    
    pipeline = Pipeline(
        [
            # generate sequential ids for users and items
            (
                cfg_pp_steps["USER_ITEM_MAPPER"],
                (
                    preprocessors.UserItemIdMapper(
                        user_id_col = pp_params["user_field"], 
                        item_id_col= pp_params["item_field"], 
                        user_id_int_col= cfg_pp_int_fields["USER_ID_INT_COL"], 
                        item_id_int_col = cfg_pp_int_fields["ITEM_ID_INT_COL"], 
                    )
                )
            ), 
            # min max scale ratings
            (
                cfg_pp_steps["TARGET_SCALER"],
                (
                    preprocessors.TargetScaler(
                        target_col = pp_params["target_field"], 
                        target_int_col = cfg_pp_int_fields["RATING_INT_COL"], 
                        prediction_int_col = cfg_pp_int_fields["PRED_RATING_INT_COL"], 
                        prediction_col = cfg_pp_int_fields["PRED_RATING_COL"], 
                        scaler_type='minmax',   # minmax, standard
                    )
                )
            ),
            # sparse matrix
            (
                cfg_pp_steps["SPARSE_MATRIX_CREATOR"],
                (
                    preprocessors.SparseMatrixCreator(
                        user_id_int_col= cfg_pp_int_fields["USER_ID_INT_COL"], 
                        item_id_int_col = cfg_pp_int_fields["ITEM_ID_INT_COL"], 
                        ratings_int_col = cfg_pp_int_fields["RATING_INT_COL"], 
                        pred_ratings_int_col = cfg_pp_int_fields["PRED_RATING_INT_COL"], 
                    )
                )
            )
        ]
    )
    
    return pipeline




def save_preprocessor(preprocess_pipe, file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    try: 
        joblib.dump(preprocess_pipe, file_path_and_name)   
    except: 
        raise Exception(f'''
            Error saving the preprocessor. 
            Does the file path exist {file_path}?''')  
    return    
    

def load_preprocessor(file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    if not os.path.exists(file_path_and_name):
        raise Exception(f'''Error: No trained preprocessor found. 
        Expected to find model files in path: {file_path_and_name}''')
        
    try: 
        preprocess_pipe = joblib.load(file_path_and_name)     
    except: 
        raise Exception(f'''
            Error loading the preprocessor. 
            Do you have the right trained preprocessor at {file_path_and_name}?''')
    
    return preprocess_pipe 



def get_inverse_transformation(preds, mask, users, pipe , model_cfg): 
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]    
    
    sparse_creator = pipe[pp_step_names["SPARSE_MATRIX_CREATOR"]]
    preds_df = sparse_creator.inverse_transform(preds, mask, users)
    
    scaler = pipe[pp_step_names["TARGET_SCALER"]]
    preds_df = scaler.inverse_transform(preds_df)

    id_mapper = pipe[pp_step_names["USER_ITEM_MAPPER"]]
    preds_df = id_mapper.inverse_transform(preds_df)
    return preds_df