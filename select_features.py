# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:44:54 2019

@author: Qiao Jiang
"""
#############################################################################################################################
#
# Import modules
#
#############################################################################################################################

# var_68 may be date

from sklearn import linear_model
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
#from sklearn.metrics import mean_squared_error
import sys
import os
import time

sys.path.append('F:\\Dropbox\\Kaggle Competition\\Santander Customer Transaction Prediction\\')
os.chdir('F:\\Dropbox\\Kaggle Competition\\Santander Customer Transaction Prediction\\')
import data_utils
import pandas as pd
import numpy as np
import seaborn as sns

import eli5
from eli5.sklearn import PermutationImportance
import pickle
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply, subtract
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.noise import GaussianNoise
from keras.layers import Bidirectional
#############################################################################################################################
#
# Load data
#
#############################################################################################################################


df_train = pd.read_csv('train.csv')

train_X, train_Y, test_X = data_utils.load_data()
X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

#############################################################################################################################
#
# Select important features, using Permutation Importance from different models
#
#############################################################################################################################

# Remove features with high correlation
# Create correlation matrix
corr_matrix = X_train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_train.drop(X_train.columns[to_drop], axis=1) # no highly correlated variables
#==============================================================================
# Check feature distribution
#==============================================================================

for i in range(3,202):
    plt.figure()
    sns.distplot(df_train.iloc[:,i], fit = norm )
    
    
sns.distplot(df_train.iloc[:,83], fit = norm )
#df_train.iloc[:,4].describe()

# possible cap or floor: var_1, var_2, var_4, var_5, 12, 16, 26, 27, 29, 35, 37,
#                        40, 41, 43, 50, 53, 59, 60, 62, 68, 69, 73, 75, 80, 81,
#                        86, 88, 89, 92, 95, 98, 99, 101, 108, 114, 123, 126, 129,
#                        134, 135, 139, 143, 145, 150, 153, 157, 158, 163, 164,
#                        165, 166, 168, 173, 175, 176, 177, 180, 181, 187, 188,
#                        191, 194, 195, 196, 197, 198                      
                         
# double peak: var_1, var_2, var_4, var_5, var_6, var_7, 8, 9, 13, 16, 18, 20
#              22, 24, 29, 30, 33, 37, 41, 43, 44, 53, 55, 59, 60, 69, 75, 76, 
#              77, 78, 79, 80, 82, 83, 85, 86, 88, 89, 90, 94, 96, 97, 99, 101,
#              108, 109, 111, 112, 113, 116, 119, 121, 123, 125, 126, 127, 129,
#              130, 131, 132, 134, 135, 136, 137, 139, 141, 145, 150, 151, 153,
#              154, 157, 158, 159, 160, 161, 163, 164, 165, 166, 168, 173, 174,
#              175, 176, 177, 180, 181, 184, 187, 188, 189, 191, 194, 196, 197,
#              199
               
# normal: var_3, 10, 11, 14, 15, 17, 19, 21, 23, 25, 26, 27, 28, 31, 32, 34, 35,
#         36, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56, 57, 58, 61, 62,
#         63, 64, 65, 66, 67, 70, 71, 72, 73, 74, 84, 87, 91, 92, 93, 95, 98, 100,
#         102, 103, 104, 105, 106, 107, 110, 114, 115, 117, 118, 120, 122, 124,
#         128, 133, 138, 140, 142, 143, 144, 146, 147, 148, 149, 152, 155, 156,
#         159, 162, 167, 169, 170, 171, 172, 178, 179, 182, 183, 185, 186, 190, 
#         192, 193, 195, 198

# possible categorical: 68

# perfect normal distribution:
# 10, 11, 17, 21, 28, 31, 38, 52, 58, 65, 67, 74, 104, 105, 106, 110, 118, 120,
# 124, 133, 138, 142, 146, 148, 149, 152, 162, 167, 172, 179, 185, 186, 190,
# 192, 193, 


# note, when some variabl is likely to be capped but there are still some values beyond that cap
#==============================================================================
# The chosen model is XGB
#==============================================================================

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train, Y_train, early_stopping_rounds=42, eval_set=[(X_valid, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.871

xgb_perm = xgb_model.feature_importances_
xgb_perm_df = pd.DataFrame({'ID_code' : X_train.columns, 'Perm':xgb_perm})
print(xgb_perm)

#sns.distplot(xgb_perm_df.iloc[:,1])

#xgb_perm_large_df = xgb_perm_df.loc[xgb_perm_df.Perm >=0.005]    
xgb_perm_large_df = xgb_perm_df.nlargest(100, 'Perm') # var_81 is very important     
xgb_perm_large_df = xgb_perm_large_df.reset_index(drop = True)
#==============================================================================
# Try LightGBM
#==============================================================================    

#==============================================================================
# Select the first 100 important features for each model and take an intersection
# to select the importnat ones for all of them
#==============================================================================

# selected raw features

train_X_nlargest = train_X.loc[:,xgb_perm_large_df.loc[:,'ID_code']]

# create more feature using +

new_feature_plus = pd.DataFrame()

for i in range(0,xgb_perm_large_df.shape[0]):
    for j in range(i+1,xgb_perm_large_df.shape[0]):
        name_temp = xgb_perm_large_df.loc[i,'ID_code'] + '_plus_' + xgb_perm_large_df.loc[j,'ID_code']
        new_feature_plus[name_temp] = train_X[xgb_perm_large_df.loc[i,'ID_code']] + train_X[xgb_perm_large_df.loc[j,'ID_code']]
        #new_feature_plus.assign(name_temp = X_train[xgb_perm_large_df.loc[i,'ID_code']] + X_train[xgb_perm_large_df.loc[j,'ID_code']])

# create more feature using -

new_feature_diff = pd.DataFrame()

for i in range(0,xgb_perm_large_df.shape[0]):
    for j in range(i+1,xgb_perm_large_df.shape[0]):
        name_temp = xgb_perm_large_df.loc[i,'ID_code'] + '_diff_' + xgb_perm_large_df.loc[j,'ID_code']
        new_feature_diff[name_temp] = train_X[xgb_perm_large_df.loc[i,'ID_code']] - train_X[xgb_perm_large_df.loc[j,'ID_code']]
        #new_feature_plus.assign(name_temp = X_train[xgb_perm_large_df.loc[i,'ID_code']] + X_train[xgb_perm_large_df.loc[j,'ID_code']])

# create more feature using *

new_feature_prod = pd.DataFrame()

for i in range(0,xgb_perm_large_df.shape[0]):
    for j in range(i+1,xgb_perm_large_df.shape[0]):
        name_temp = xgb_perm_large_df.loc[i,'ID_code'] + '_prod_' + xgb_perm_large_df.loc[j,'ID_code']
        new_feature_prod[name_temp] = train_X[xgb_perm_large_df.loc[i,'ID_code']] * train_X[xgb_perm_large_df.loc[j,'ID_code']]
        #new_feature_plus.assign(name_temp = X_train[xgb_perm_large_df.loc[i,'ID_code']] + X_train[xgb_perm_large_df.loc[j,'ID_code']])

# create more feature using *

new_feature_divi = pd.DataFrame()

for i in range(0,xgb_perm_large_df.shape[0]):
    for j in range(i+1,xgb_perm_large_df.shape[0]):
        name_temp = xgb_perm_large_df.loc[i,'ID_code'] + '_divide_' + xgb_perm_large_df.loc[j,'ID_code']
        new_feature_divi[name_temp] = train_X[xgb_perm_large_df.loc[i,'ID_code']] / train_X[xgb_perm_large_df.loc[j,'ID_code']]
        #new_feature_plus.assign(name_temp = X_train[xgb_perm_large_df.loc[i,'ID_code']] + X_train[xgb_perm_large_df.loc[j,'ID_code']])


#==============================================================================
# Select the first 100 important features for generation and take an intersection
# to select the importnat ones for all of them
#==============================================================================

train_X_new_plus = pd.concat([train_X_nlargest, new_feature_plus],axis = 1, ignore_index = True)
train_X_new_diff = pd.concat([train_X_nlargest, new_feature_diff],axis = 1, ignore_index = True)
train_X_new_prod = pd.concat([train_X_nlargest, new_feature_prod],axis = 1, ignore_index = True)
train_X_new_divi = pd.concat([train_X_nlargest, new_feature_divi],axis = 1, ignore_index = True)

# select 100 from plus

# first, get score with raw feature

X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(train_X_new_plus, train_Y, test_size=0.2, random_state=42)

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.822

xgb_perm = xgb_model.feature_importances_

name_all_temp = train_X_nlargest.columns.values.tolist() + new_feature_plus.columns.values.tolist() # need to be changed

xgb_perm_df = pd.DataFrame({'ID_code' : name_all_temp, 'Perm':xgb_perm})
print(xgb_perm)

raw_feature_temp = list(set(list(xgb_perm_df.ID_code)) & set(train_X.columns))

xgb_perm_df = xgb_perm_df[~xgb_perm_df.ID_code.isin(raw_feature_temp)]

xgb_perm_large_df_plus_with = xgb_perm_df.nlargest(400, 'Perm') # var_81 is very important     
xgb_perm_large_df_plus_with = xgb_perm_large_df_plus_with.reset_index(drop = True)

# second, get score without raw feature

X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(new_feature_plus, train_Y, test_size=0.2, random_state=42)

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.824

xgb_perm = xgb_model.feature_importances_

#name_all_temp = train_X_nlargest.columns.values.tolist() + new_feature_plus.columns.values.tolist()
xgb_perm_df = pd.DataFrame({'ID_code' : new_feature_plus.columns.values.tolist(), 'Perm':xgb_perm}) # need to be changed
#raw_feature_temp = list(set(list(xgb_perm_df.ID_code)) & set(train_X.columns))
#xgb_perm_df = xgb_perm_df[~xgb_perm_df.ID_code.isin(raw_feature_temp)]

xgb_perm_large_df_plus_without = xgb_perm_df.nlargest(400, 'Perm') # var_81 is very important     
xgb_perm_large_df_plus_without = xgb_perm_large_df_plus_without.reset_index(drop = True)

# select 100 from diff

# first, get score with raw feature

X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(train_X_new_diff, train_Y, test_size=0.2, random_state=42)

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.0.820

xgb_perm = xgb_model.feature_importances_

name_all_temp = train_X_nlargest.columns.values.tolist() + new_feature_diff.columns.values.tolist() # need to be changed

xgb_perm_df = pd.DataFrame({'ID_code' : name_all_temp, 'Perm':xgb_perm})
print(xgb_perm)

raw_feature_temp = list(set(list(xgb_perm_df.ID_code)) & set(train_X.columns))

xgb_perm_df = xgb_perm_df[~xgb_perm_df.ID_code.isin(raw_feature_temp)]

xgb_perm_large_df_diff_with = xgb_perm_df.nlargest(400, 'Perm') # var_81 is very important    # need to be changed 
xgb_perm_large_df_diff_with = xgb_perm_large_df_diff_with.reset_index(drop = True) # need to be changed

# second, get score without raw feature

X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(new_feature_diff, train_Y, test_size=0.2, random_state=42)

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.815

xgb_perm = xgb_model.feature_importances_

xgb_perm_df = pd.DataFrame({'ID_code' : new_feature_diff.columns.values.tolist(), 'Perm':xgb_perm}) # need to be changed
#raw_feature_temp = list(set(list(xgb_perm_df.ID_code)) & set(train_X.columns))
#xgb_perm_df = xgb_perm_df[~xgb_perm_df.ID_code.isin(raw_feature_temp)]

xgb_perm_large_df_diff_without = xgb_perm_df.nlargest(400, 'Perm') # var_81 is very important     
xgb_perm_large_df_diff_without = xgb_perm_large_df_diff_without.reset_index(drop = True)

# select 100 from prod

# first, get score with raw feature

X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(train_X_new_prod, train_Y, test_size=0.2, random_state=42)

del train_X_new_prod

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.824

xgb_perm = xgb_model.feature_importances_

name_all_temp = train_X_nlargest.columns.values.tolist() + new_feature_prod.columns.values.tolist() # need to be changed

xgb_perm_df = pd.DataFrame({'ID_code' : name_all_temp, 'Perm':xgb_perm})
print(xgb_perm)

raw_feature_temp = list(set(list(xgb_perm_df.ID_code)) & set(train_X.columns))

xgb_perm_df = xgb_perm_df[~xgb_perm_df.ID_code.isin(raw_feature_temp)]

xgb_perm_large_df_prod_with = xgb_perm_df.nlargest(400, 'Perm') # var_81 is very important    # need to be changed 
xgb_perm_large_df_prod_with = xgb_perm_large_df_prod_with.reset_index(drop = True) # need to be changed

# second, get score without raw feature

X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(new_feature_prod, train_Y, test_size=0.2, random_state=42)

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.818

xgb_perm = xgb_model.feature_importances_

xgb_perm_df = pd.DataFrame({'ID_code' : new_feature_prod.columns.values.tolist(), 'Perm':xgb_perm}) # need to be changed
#raw_feature_temp = list(set(list(xgb_perm_df.ID_code)) & set(train_X.columns))
#xgb_perm_df = xgb_perm_df[~xgb_perm_df.ID_code.isin(raw_feature_temp)]

xgb_perm_large_df_prod_without = xgb_perm_df.nlargest(400, 'Perm') # var_81 is very important     
xgb_perm_large_df_prod_without = xgb_perm_large_df_prod_without.reset_index(drop = True)

# select 100 from divi

# first, get score with raw feature

X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(train_X_new_divi, train_Y, test_size=0.2, random_state=42)

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.824

xgb_perm = xgb_model.feature_importances_

name_all_temp = train_X_nlargest.columns.values.tolist() + new_feature_divi.columns.values.tolist() # need to be changed

xgb_perm_df = pd.DataFrame({'ID_code' : name_all_temp, 'Perm':xgb_perm})
print(xgb_perm)

raw_feature_temp = list(set(list(xgb_perm_df.ID_code)) & set(train_X.columns))

xgb_perm_df = xgb_perm_df[~xgb_perm_df.ID_code.isin(raw_feature_temp)]

xgb_perm_large_df_divi_with = xgb_perm_df.nlargest(400, 'Perm') # var_81 is very important    # need to be changed 
xgb_perm_large_df_divi_with = xgb_perm_large_df_divi_with.reset_index(drop = True) # need to be changed

# second, get score without raw feature

X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(new_feature_divi, train_Y, test_size=0.2, random_state=42)

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.5)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.818

xgb_perm = xgb_model.feature_importances_

xgb_perm_df = pd.DataFrame({'ID_code' : new_feature_divi.columns.values.tolist(), 'Perm':xgb_perm}) # need to be changed
#raw_feature_temp = list(set(list(xgb_perm_df.ID_code)) & set(train_X.columns))
#xgb_perm_df = xgb_perm_df[~xgb_perm_df.ID_code.isin(raw_feature_temp)]

xgb_perm_large_df_divi_without = xgb_perm_df.nlargest(400, 'Perm') # var_81 is very important     
xgb_perm_large_df_divi_without = xgb_perm_large_df_divi_without.reset_index(drop = True)

#==============================================================================
# Select the most important features.
#==============================================================================

all_features = pd.concat([train_X, new_feature_plus], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_diff], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_prod], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_divi], axis = 1, ignore_index = True)

col_names = list(set(list(xgb_perm_large_df_diff_with.ID_code)) & set(list(xgb_perm_large_df_diff_without.ID_code)))
col_names = list(set(col_names)|set(list(xgb_perm_large_df_divi_with.ID_code)) & set(list(xgb_perm_large_df_divi_without.ID_code)))
col_names = list(set(col_names)|set(list(xgb_perm_large_df_prod_with.ID_code)) & set(list(xgb_perm_large_df_prod_without.ID_code)))
col_names = list(set(col_names)|set(list(xgb_perm_large_df_plus_with.ID_code)) & set(list(xgb_perm_large_df_plus_without.ID_code)))
col_names = list(set(col_names)|set(train_X.columns))

# 634 features

new_feature_plus_light = new_feature_plus.iloc[:,new_feature_plus.columns.isin(col_names)]
new_feature_prod_light = new_feature_plus.iloc[:,new_feature_prod.columns.isin(col_names)]
new_feature_divi_light = new_feature_plus.iloc[:,new_feature_divi.columns.isin(col_names)]
new_feature_diff_light = new_feature_plus.iloc[:,new_feature_diff.columns.isin(col_names)]

all_features = pd.concat([train_X, new_feature_plus_light], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_diff_light], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_prod_light], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_divi_light], axis = 1, ignore_index = True)

train_X_new = all_features
train_X_new.columns = col_names
df_train_new = pd.DataFrame({'ID_code' : df_train['ID_code'],'target' : train_Y})
df_train_new = pd.concat([df_train_new,train_X_new],axis = 1)

df_train_new.to_csv('F:\\Dropbox\\Kaggle Competition\\Santander Customer Transaction Prediction\\train_new.csv', index = False)
#==============================================================================
# Construct the new test set
#==============================================================================

new_feature_plus = pd.DataFrame()

for i in range(0,xgb_perm_large_df.shape[0]):
    for j in range(i+1,xgb_perm_large_df.shape[0]):
        name_temp = xgb_perm_large_df.loc[i,'ID_code'] + '_plus_' + xgb_perm_large_df.loc[j,'ID_code']
        new_feature_plus[name_temp] = test_X[xgb_perm_large_df.loc[i,'ID_code']] + test_X[xgb_perm_large_df.loc[j,'ID_code']]
        #new_feature_plus.assign(name_temp = X_train[xgb_perm_large_df.loc[i,'ID_code']] + X_train[xgb_perm_large_df.loc[j,'ID_code']])

# create more feature using -

new_feature_diff = pd.DataFrame()

for i in range(0,xgb_perm_large_df.shape[0]):
    for j in range(i+1,xgb_perm_large_df.shape[0]):
        name_temp = xgb_perm_large_df.loc[i,'ID_code'] + '_diff_' + xgb_perm_large_df.loc[j,'ID_code']
        new_feature_diff[name_temp] = test_X[xgb_perm_large_df.loc[i,'ID_code']] - test_X[xgb_perm_large_df.loc[j,'ID_code']]
        #new_feature_plus.assign(name_temp = X_train[xgb_perm_large_df.loc[i,'ID_code']] + X_train[xgb_perm_large_df.loc[j,'ID_code']])

# create more feature using *

new_feature_prod = pd.DataFrame()

for i in range(0,xgb_perm_large_df.shape[0]):
    for j in range(i+1,xgb_perm_large_df.shape[0]):
        name_temp = xgb_perm_large_df.loc[i,'ID_code'] + '_prod_' + xgb_perm_large_df.loc[j,'ID_code']
        new_feature_prod[name_temp] = test_X[xgb_perm_large_df.loc[i,'ID_code']] * test_X[xgb_perm_large_df.loc[j,'ID_code']]
        #new_feature_plus.assign(name_temp = X_train[xgb_perm_large_df.loc[i,'ID_code']] + X_train[xgb_perm_large_df.loc[j,'ID_code']])

# create more feature using *

new_feature_divi = pd.DataFrame()

for i in range(0,xgb_perm_large_df.shape[0]):
    for j in range(i+1,xgb_perm_large_df.shape[0]):
        name_temp = xgb_perm_large_df.loc[i,'ID_code'] + '_divide_' + xgb_perm_large_df.loc[j,'ID_code']
        new_feature_divi[name_temp] = test_X[xgb_perm_large_df.loc[i,'ID_code']] / test_X[xgb_perm_large_df.loc[j,'ID_code']]
        #new_feature_plus.assign(name_temp = X_train[xgb_perm_large_df.loc[i,'ID_code']] + X_train[xgb_perm_large_df.loc[j,'ID_code']])

new_feature_plus_light = new_feature_plus.iloc[:,new_feature_plus.columns.isin(col_names)]
new_feature_prod_light = new_feature_plus.iloc[:,new_feature_prod.columns.isin(col_names)]
new_feature_divi_light = new_feature_plus.iloc[:,new_feature_divi.columns.isin(col_names)]
new_feature_diff_light = new_feature_plus.iloc[:,new_feature_diff.columns.isin(col_names)]

all_features = pd.concat([test_X, new_feature_plus_light], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_diff_light], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_prod_light], axis = 1, ignore_index = True)
all_features = pd.concat([all_features, new_feature_divi_light], axis = 1, ignore_index = True)

test_X_new = all_features
test_X_new.columns = col_names

test_for_ID_code = pd.read_csv('F:\\Dropbox\\Kaggle Competition\\Santander Customer Transaction Prediction\\test.csv')


test_new = pd.DataFrame({'ID_code' : test_for_ID_code['ID_code']})
test_new = pd.concat([test_new,test_X_new],axis = 1)
test_new.to_csv('F:\\Dropbox\\Kaggle Competition\\Santander Customer Transaction Prediction\\test_new.csv', index = False)

#==============================================================================


X_train_nlargest, X_valid_nlargest, Y_train, Y_valid = train_test_split(train_X_new, train_Y, test_size=0.2, random_state=42)

start = time.process_time()

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
xgb_model.fit(X_train_nlargest, Y_train, early_stopping_rounds=42, eval_set=[(X_valid_nlargest, Y_valid)], verbose=False)

print(time.process_time() - start)

predictions = xgb_model.predict_proba(X_valid_nlargest)[:,1]
result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})
fpr, tpr, thresholds = roc_curve(result_df['ground_truth'], result_df['result'], pos_label=1)
xgb_valid_auc = auc(fpr, tpr) # 0.871





















