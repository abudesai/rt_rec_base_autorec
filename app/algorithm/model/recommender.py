
import numpy as np, pandas as pd
import os, sys
from sklearn.utils import shuffle
import joblib

import os, warnings, sys 
warnings.filterwarnings('ignore') 


import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


MODEL_NAME = "recommender_base_autorecommender"

model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"



COST_THRESHOLD = float('inf')



class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("\nCost is inf, so stopping training!!")
            self.model.stop_training = True


class Recommender():
    def __init__(self, M, drop_out=0.7, l2_reg=0., lr = 0.1, batch_size = 256, momentum = 0.95, **kwargs  ):
        self.M = M
        self.drop_out = drop_out 
        self.l2_reg = l2_reg
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.mu = 0.

        self.model = self.build_model()

        def custom_loss(y_true, y_pred):
            mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
            diff = y_pred - y_true
            sqdiff = diff * diff * mask
            sse = K.sum(K.sum(sqdiff))
            n = K.sum(K.sum(mask))
            return sse / n
            

        self.model.compile(
            loss=custom_loss,
            # optimizer=Adam(lr=0.01),
            optimizer=SGD(learning_rate=self.lr, momentum=self.momentum),
            metrics=[custom_loss],
        )


    def build_model(self): 
        # build the model - just a 1 hidden layer autoencoder
        i = Input(shape=(self.M,))
        x = Dropout(self.drop_out)(i)
        num_hidden = min(self.M // 10, 50)
        x = Dense(num_hidden, activation='tanh', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dense(self.M, kernel_regularizer=l2(self.l2_reg))(x)
        model = Model(i, x)
        return model 


    def fit(self, train_data_tup, valid_data_tup, epochs=100, verbose=0): 
        batch_size = self.batch_size
        def generator(X_R, X_M, Y_R, Y_M):
            while True:
                if X_R.shape[0] % batch_size == 0:
                    num_batches = X_R.shape[0] // batch_size
                else:
                    num_batches = X_R.shape[0] // batch_size + 1

                for i in range(num_batches ):
                    upper = min((i+1)*batch_size, X_M.shape[0])
                    x_r = X_R[i*batch_size:upper].toarray()
                    x_m = X_M[i*batch_size:upper].toarray()

                    y_r = Y_R[i*batch_size:upper].toarray()
                    y_m = Y_M[i*batch_size:upper].toarray()

                    x_r = x_r - self.mu * x_m
                    y_r = y_r - self.mu * y_m

                    yield x_r, y_r  # returns X and Y

        train_X_R, train_X_M, train_Y_R, train_Y_M = train_data_tup
        if valid_data_tup is not None: 
            valid_X_R, valid_X_M, valid_Y_R, valid_Y_M = valid_data_tup
            validation_data = generator(valid_X_R, valid_X_M, valid_Y_R, valid_Y_M)
        else:
            validation_data = None

        self.mu = train_X_R.sum() / train_X_M.sum()
        # print("mu: ", self.mu)

        early_stop_loss = 'val_loss' if validation_data != None else 'loss'
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-4, patience=3) 
        infcost_stop_callback = InfCostStopCallback()


        history = self.model.fit(
                x=generator(train_X_R, train_X_M, train_Y_R, train_Y_M),
                validation_data=validation_data,
                batch_size = batch_size,
                epochs=epochs,
                steps_per_epoch=train_X_R.shape[0] // batch_size + 1,
                validation_steps=valid_X_R.shape[0] // batch_size + 1 if valid_data_tup is not None else None,
                verbose=verbose,
                shuffle=True,
                callbacks=[early_stop_callback, infcost_stop_callback]
            )            
        return history


    def predict(self, data, data_mask):
        R = data - self.mu * data_mask
        preds = self.model.predict(R, batch_size=100, verbose=1) + self.mu
        return preds 
    
    def evaluate(self, X, X_mask, y_true): 
        R = X - self.mu * X_mask
        y_pred = self.model.predict(R, batch_size=100, verbose=1) + self.mu
        
        mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
        mask = y_true != 0
        diff = y_pred - y_true
        sqdiff = diff * diff * mask
        sse = sqdiff.sum()
        mse = sse / mask.sum()
        return mse
        

    def summary(self):
        self.model.summary()

    def save(self, model_path): 
        model_params = {
            "M": self.M,
            "drop_out": self.drop_out,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "mu": self.mu,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))

        self.model.save_weights(os.path.join(model_path, model_wts_fname))


    @staticmethod
    def load(model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        rec_model = Recommender(**model_params)
        rec_model.mu = model_params['mu']
        rec_model.model.load_weights(os.path.join(model_path, model_wts_fname)).expect_partial()
        return rec_model


def get_data_based_model_params(R): 
    M = R.shape[1]
    return {"M": M}



def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = Recommender.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path): 
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(f_path, history_fname)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f, indent=2)
