# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:00:58 2019

@author: fbrev

based on visually-impaired-aid-cv
transfer learning version

Required packages:
    tensorflow
    pandas
    scikit-learn
    pillow
    tensorflow-addons
"""

import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
import os
import random
#print(os.listdir("../via-dataset/images/"))

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

DATASET_PATH = "../via-dataset/images/" 

def load_data():
    
    filenames = os.listdir(DATASET_PATH)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'clear':
            categories.append(0)
        else:
            categories.append(1)
    
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df

def ensembles_single_model():
   
    model_list = ['MobileNet','MobileNet','EfficientNetB0','EfficientNetB4','InceptionV3','MobileNet']
    conf_list = ['A','B','C','D','E','F']

    # matrix to save the scores   
    ens_res = []
  
    # For each model
    for index, model in enumerate(model_list):
        conf = conf_list[index]
        
        acc_table = []
        # For each split
        for split, [train, test] in enumerate(kf.split(df)):
            
            test_df = df.loc[test]  
            y_true = test_df["category"]
            
            acc_line = []
            predictions = []        
            
            for inst in range(1,11):
                # Load the predictions
                pred_filename = "./pred_ensemble/conf" + conf + "/pred-" + model + "-split=" + str(split+1) + "-inst=" + str(inst) + ".csv"
                predictions.append(np.loadtxt(pred_filename, delimiter=','))            
                # Get the average predictions
                mean_pred = np.mean(predictions,0)
                y_pred = np.argmax(mean_pred,1)
                # calculate the confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                acc = np.trace(cm) / np.sum(cm) #accuracy
                acc_line.append(acc)
                           
            acc_table.append(acc_line)
    
        ens_res.append(np.mean(acc_table,0))
        
    # Transpose the output matrix to fit the article
    ens_res = np.transpose(ens_res) 
              
    # Save the accuracy to a CSV file.
    ens_res_filename = './ensemble-single-res.csv'
    os.makedirs(os.path.dirname(ens_res_filename), exist_ok=True)
    np.savetxt(ens_res_filename, ens_res, delimiter=',', fmt='%0.4f')    
  

def ensembles_multi_model():
    
    model_list = ['EfficientNetB4','EfficientNetB0','MobileNet','MobileNet','MobileNet','InceptionV3']
    conf_list = ['D','C','A','B','F','E']
    
    ens_res = []
    
    for instcount in range(1,11):
        acc_table = []
        for split, [train, test] in enumerate(kf.split(df)):    
            predictions = []
            acc_line = []
            test_df = df.loc[test]  
            y_true = test_df["category"]
            for index, model in enumerate(model_list):
                for inst in range(1,instcount+1):
                    conf = conf_list[index]
                    # Load the predictions
                    pred_filename = "./pred_ensemble/conf" + conf + "/pred-" + model + "-split=" + str(split+1) + "-inst=" + str(inst) + ".csv"
                    predictions.append(np.loadtxt(pred_filename, delimiter=','))                
    
                # Get the average predictions
                mean_pred = np.mean(predictions,0)
                y_pred = np.argmax(mean_pred,1)
                # calculate the confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                acc = np.trace(cm) / np.sum(cm) #accuracy            
                acc_line.append(acc)
        
            acc_table.append(acc_line)
        ens_res.append(np.mean(acc_table,0))
                  
    # Save the accuracy to a CSV file.
    ens_res_filename = './ensemble-multi-res.csv'
    os.makedirs(os.path.dirname(ens_res_filename), exist_ok=True)
    np.savetxt(ens_res_filename, ens_res, delimiter=',', fmt='%0.4f')       

# Main
if __name__ == "__main__":
    
    df = load_data()
    
    # creating folds for cross-validation
    from sklearn.model_selection import RepeatedKFold
    kfold_n_splits = 10
    kfold_n_repeats = 5
    kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats, random_state=SEED)
    kf.split(df)         
       
    ensembles_single_model()
    ensembles_multi_model()