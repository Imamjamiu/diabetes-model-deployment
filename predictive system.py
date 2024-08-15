# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import pickle


#loading the save model
loaded_model = pickle.load(open("C:/Users/NUGGET/Desktop\model deployment/diabetes_trained_model.sav","rb"))
input_data = (10,125,70,26,115,31.1,0.205,41)
# changing the input data into numpy array

input_data_as_numpy_array = np.asarray(input_data)
# reshaping the array because we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# stanndardizing the input data

prediction = loaded_model.predict(input_data_reshaped)

print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
    