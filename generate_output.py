import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from catboost import *

data_test = pd.read_csv('test_features.csv')

final_data=pd.read_csv('train_features.csv')

X_train=final_data.drop(['id','target'],axis=1)
y_train=final_data['target']
X_test=data_test.drop('id',axis=1)
X_test=X_test.values

cgb = CatBoostRegressor(learning_rate=0.3,verbose=100,use_best_model=False,task_type="GPU",loss_function='RMSE',iterations=20000)

cgb.fit(X_train,y_train)

p=cgb.predict(X_test)

id = np.array(data_test['id'])

sub = np.concatenate((np.reshape(id, (1000,1)),np.reshape(p, (1000,1))), axis = 1)

pd.DataFrame(sub).to_csv('output.csv',index=False)