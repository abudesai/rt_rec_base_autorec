import numpy as np, pandas as pd
import os, sys

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.recommender as recommender


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.user_id = data_schema["inputDatasets"]["recommenderBaseMainInput"]["userField"]   
        self.item_id = data_schema["inputDatasets"]["recommenderBaseMainInput"]["itemField"]   
        
        
    
    def _get_preprocessor(self): 
        try: 
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
            return self.preprocessor
        except: 
            print(f'No preprocessor found to load from {self.model_path}. Did you train the model first?')
        return None
    
    
    def _get_model(self): 
        try: 
            self.model = recommender.load_model(self.model_path)
            return self.model
        except: 
            print(f'No model found to load from {self.model_path}. Did you train the model first?')
        return None
    
        
    
    # def predict(self, data, data_schema):  
        
    #     preprocessor = self._get_preprocessor()
    #     model = self._get_model()
        
    #     if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
    #     if model is None:  raise Exception("No model found. Did you train first?")
        
    #     # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
    #     proc_data = preprocessor.transform(data)  
    #     # Grab input features for prediction
    #     pred_X = proc_data['X']
    #     ids = proc_data['ids']
    #     # make predictions
    #     preds = model.predict( pred_X )
    #     # inverse transform the predictions to original scale
    #     preds = pipeline.get_inverse_transform_on_preds(preprocessor, model_cfg, preds) 
    #     # get the names for the id and prediction fields
    #     id_field_name = data_schema["inputDatasets"]["recommenderBaseMainInput"]["idField"]     
    #     # return the prediction df with the id and prediction fields
    #     preds_df = pd.DataFrame(np.round(preds,4), columns=['prediction'])
    #     # add the id field to the dataframe
    #     preds_df.insert(0, id_field_name, ids)        
    #     return preds_df
    

    
    
    def predict(self, data):  
        
        print("Running predictions...")
        
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
        
        N = data.shape[0]
        max_batch_size = model_cfg["max_batch_size"]
        num_batches = (N // max_batch_size) if N % max_batch_size == 0 else (N // max_batch_size) + 1
    
        
        all_preds = []
        for i in range(num_batches): 
            
            mini_batch = data.iloc[i*max_batch_size : (i+1)*max_batch_size, :]

            # transform data
            test_X_R, test_X_M, test_Y_R, test_Y_M, test_users_int_id = preprocessor.transform(mini_batch)
            if test_X_R is None or test_X_R is None: continue

            # print('processed train data and mask shape:',  test_X_R.shape, test_X_M.shape, test_Y_R.shape, test_Y_M.shape)
        
            # make predictions
            preds = model.predict(test_X_R, test_X_M )
            
            # make inverse transformations on predictions
            preds_df = pipeline.get_inverse_transformation(preds, test_Y_M, test_users_int_id, preprocessor, model_cfg)

            preds_df = mini_batch.merge(
                preds_df[[self.user_id, self.item_id, model_cfg["pp_params"]["int_fields"]["PRED_RATING_COL"]]], 
                on=[self.user_id, self.item_id])   
            
            all_preds.append(preds_df)
        
        if len(all_preds) == 0:
            msg = '''
            Pre-processed prediction data is empty. No predictions to run.
            This usually occurs if none of the users and/or items in prediction data
            were present in the training data. 
            '''
            print(msg)
            return None
        else: 

            all_preds = pd.concat(all_preds, ignore_index=True)
        
        return all_preds
        
        

