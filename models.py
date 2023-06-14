from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential

class Models:

    def import_model(self,model_name):
        display_func = getattr(self, model_name)
        
        return display_func()
        
    
    def model2_bpm(self):

        dropout = 0.2
        input_size = 60

        model = keras.Sequential()
        model.add(LSTM(input_size, return_sequences=True,input_shape=(input_size, 1)))
        model.add(Dropout(rate=dropout))
        model.add(Bidirectional(LSTM((input_size * 2), return_sequences=True))) 
        model.add(Dropout(rate=dropout))
        model.add(Bidirectional(LSTM(input_size, return_sequences=False))) 
        model.add(Dense(units=2))
        model.add(Activation('relu'))
        
        return model
    
    def model2_hr(self):
        
        dropout = 0.2
        input_size = 60

        model = keras.Sequential()
        model.add(LSTM(input_size, return_sequences=True,input_shape=(input_size, 1)))
        model.add(Dropout(rate=dropout))
        model.add(Bidirectional(LSTM((input_size * 2), return_sequences=True))) 
        model.add(Dropout(rate=dropout))
        model.add(Bidirectional(LSTM(input_size, return_sequences=False))) 
        model.add(Dense(units=1))
        model.add(Activation('linear'))
        
        return model
    

    