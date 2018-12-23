# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:16:36 2018

@author: rciszek
"""
from __future__ import unicode_literals

from read_data import read_EPIBIOS_data,filterMissing

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from fancyimpute import MICE
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score

def MEICC(data, group_column, value_columns):
    '''
    Calculates multinomal intraclass correlation using euclidean distance measure
    
    # Arguments:
        data : DataFrame of protocol data.
        group_column : Name of the column defining the grouping of analyzed columns.
        columns : Column names of the features included in protocol feature vector
        
    # Return:
        mICC : Multinomal ICC
    '''    
    
    unique_groups = np.unique(data[group_column])
    column_data = data[value_columns]
    n = data.shape[0]
    k = unique_groups.shape[0]
    
    if k == 1:
        return np.nan
    
    group_means = np.zeros((k,column_data.shape[1]))
    wgds = 0
    bgds = 0
    n_g_squared = 0
    grand_mean = np.mean(column_data,axis=0)

    for i, group in enumerate(unique_groups):
        group_values = data.loc[data[group_column] == group][value_columns].values

        n_group = group_values.shape[0]
        n_g_squared += np.power(n_group,2)

        group_means[i,:] = np.mean(group_values,axis=0)

        bgds += n_group*np.power(distance.euclidean(group_means[i,:],grand_mean),2)        
        for j in range(0,n_group):         
            wgds += np.power(distance.euclidean(group_means[i,:],group_values[j,:]),2)            
    
    wgds = wgds / (n-1 - k-1)
    bgds = bgds / (k-1)

    nA = (1 / (k-1))*(n - n_g_squared/n)

    return np.max( (( bgds - wgds) / ( (nA-1)*wgds + bgds ),0) ) 
    

def confidence_from_bootsrap(results, interval=0.05):
    '''
    Calculates upper and lower confidence interval from a given set of bootsrap
    samples.
    
    # Arguments:
        results : Array of bootsrap sampling results
        interval : Target confidence interval. Default: 0.05
        
    # Return:
        lower_ci : Lower confidence interval
        upper_ci : Upper confidence interval
        
    '''
    results = np.sort(results)
    results = results[~np.isnan(results)]
    
    if results.shape[0] == 0:
        return np.nan, np.nan
    
    cutoff = (interval/2.0)
    lower_ci = results[int(cutoff*results.shape[0])]
    upper_ci = results[int((1-cutoff)*results.shape[0])]

    return lower_ci, upper_ci
        

def AUC_assessment(data, group_column, 
                   value_columns, impute=False, 
                   n_neighbors = 10, 
                   n_jobs = 6):
    '''
    Assessess the level of harmonization with micro averaged AUC score.
    
    # Arguments:
        data : Protocol data as DataFrame.
        group_column : Column used to group the observations, e.g. the column 
            containing center labels.
        value_columns : List of column names denoting the features included in
            the protocol feature vector.
        impute : If False, observations with missing values will be omitted. If
            True, missing values will be imputed using MICE.
        n_neighbors : Number of neighbors utiled in KNN calculations.
        n_jobs : Number of parallel jobs run in KNN
    # Return :    
          auc_score : Micro averaged AUC score
    '''
    
    n_groups = np.unique(data[group_column].values).shape[0]
    loo = LeaveOneOut()
    predicted_proba = np.zeros((data.shape[0], 1))
    actual = np.zeros((data.shape[0],1))
    
    
    #Encode centers numerically by default.
    le = LabelEncoder()
    le.fit(data[group_column].values)
    y = le.transform(data[group_column].values)
    n_classes = np.unique(y).shape[0]
    
    #If only single class is present, abort and return-1
    if n_classes == 1:
        return -1
    #If more than two centers are present in the data, use one-hot encoding instead.
    if n_classes > 2:
        lb = LabelBinarizer()
        lb.fit(y.tolist())
        y = lb.transform(y)
        y = np.array(y)
        predicted_proba = np.zeros((data.shape[0], n_groups))
        actual = np.zeros((data.shape[0],n_groups))    

    if y.shape[0] == 0:
        return np.nan

    x = data[ value_columns ].values
    
    if impute:
        x = MICE(verbose=0).complete(x)

    if y.ndim ==1:
        y = np.expand_dims(y,1)
        
    sample_index = 0
    for train_i,test_i in loo.split(x):

        scaler = StandardScaler()
        x_train = x[train_i,:]
        y_train = y[train_i,:]
        x_test = x[test_i,:]
               
        scaler.fit(x_train)
        
        #Standardize the test set using the std and mean of the training set    
        x_split = scaler.transform(x_train)    
        test_split = scaler.transform(x_test)  
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors,n_jobs=n_jobs)
       
        neigh.fit(x_split, y_train)
        proba = neigh.predict_proba(test_split)
        if len(proba) == 1:
             predicted_proba[sample_index,0] = proba[0][1]
        else:
            proba = [ entry[0,-1] for entry in proba]
            predicted_proba[sample_index,:] = proba[:]

        actual[sample_index,:] = y[test_i,:]
        sample_index += 1

    if n_classes > 2:
        auc_score = roc_auc_score(actual,predicted_proba,average='micro')
    else:
        auc_score = roc_auc_score(y,predicted_proba[:,1])
    
    return auc_score
  
def filtered_group_harmonization_assessment(filter_column, 
                                  filter_value, 
                                  group_column, 
                                  feature_subsets, 
                                  column_set, data,
                                  impute=False, 
                                  sample=False,
                                  scoring='AUC'):
    '''
    Assesses  the differentiability of a specific subgroup within centers in 
    terms of classification AUC or ICC.
    
    # Arguments
        filter_column : Name of the column for filtering subjects by a specific condition
            (e.g. 'treatment_group')
        filter_value : Value of the filtering column denoting included subjects.
            (e.g. TBI)
        group_column : Column used to group the subjecets
        feature_subsets : List of feature subset names (strings) defining the
            feature subsets included in the analysis.
        column_set : Dictionary mapping feature set names to column names
        data : Protocol data as a DataFrame.
        impute: If True, missing values are imputed using MICE. If False, rows
            with missing values are omitted.
        sample : If True, observations are sampled with replacement.
        method : AUC or ICC
        
    # Return:
        score : KNN classification AUC or mICC.
    '''
    scores = np.zeros((1,len(feature_subsets)))
    # Loop through all selected feature sets
    for c_i, colum_set_key in enumerate(feature_subsets):
       
        if colum_set_key == 'id_data':
            continue
        
        column_names = column_set[colum_set_key]

        data_subset = data.copy()     
    
        if filter_value != None:
            data_subset = data.loc[ data[filter_column] == filter_value ]
 
        data_subset = filterMissing(data_subset, ['Location'] + column_names, colum_set_key)    
        
        if not impute:
            data_subset = data_subset.dropna(axis=0,how='any')
        elif scoring == 'ICC':
            data_subset[column_names] = MICE(verbose=0).complete(data_subset[column_names])

        if data_subset.shape[0] == 0:
            continue            
            
        # If the function is called as a part of bootstrapping loop, setting 
        # argument 'sample' to True allows scores to be calculated on subjects
        # sampled with replacement
        if sample == True:
            fn = lambda obj: obj.loc[np.random.choice(obj.index, obj.index.shape[0], replace=True),:]
            data_subset = data_subset.groupby('Location', as_index=False).apply(fn)  

        if scoring == 'AUC':
            roc_auc = AUC_assessment(data_subset, group_column, column_names,impute=impute)
            scores[0,c_i] = roc_auc
        else:
            data_subset[column_names] = StandardScaler().fit_transform(data_subset[column_names])
           
            icc = MEICC(data_subset,'Location',column_names)
            scores[0,c_i] = icc          
    return scores

    
def create_harmonization_report(protocol_data,
                                feature_sets,
                                treatment_groups = 'treatment_group',
                                impute=False, 
                                file_name="ICC.csv", 
                                repeats=1000, 
                                scoring='ICC',
                                selected_sets =  ['Combined neuroscores','Neuroscore baseline','Neuroscore 2','Neuroscore 7','Neuroscore 14','Neuroscore 21','Neuroscore 28','Lateral FPI','Blood absorbance','Weight','Distance between timepoints','Procedure timing', 'All protocol variables'],
                                ):
    '''
    Calculates ICC or AUC scores for specified feature sets and saves the results in CSV format
    
    # Arguments
        protocol_data : Protocol data as DataFrame. 
        feature_sets : Dictionary which maps feature set names to columns of the
            protocol_data.
        treatment_groups : Different group column.             
        impute : If yes, missing values will be omitted using MICE, otherwise
            rows with missing entries will be omitted.
        file_name : Result CSV file name
        repeats : Number of bootstrap iterations
        scoring : Method used to assess harmonization, Either 'AUC' or 'ICC'.
            By default, 'ICC'.
        selected_sets : Names of the feature sets included in the assessment. By 
            default, all EpiBios4rx feature sets are included.       
    '''
    
    #Analyze all individual treatmen groups and combination of all groups
    treatment_groups = protocol_data[treatment_groups].dropna().unique().tolist()
    treatment_group_names = treatment_groups + ['+'.join(treatment_groups)]
    
    n_treatment_groups = len(treatment_group_names)
    n_selected_sets = len(selected_sets)
    report_columns =  ['Feature set','mean', 'Lower CI', 'Upper CI']
    report_columns = [  i +' ' + j for j in treatment_group_names   for i in ['mean', 'lower CI', 'upper CI']]    
     
    bootstrap_scores = np.zeros((n_treatment_groups,repeats, len(selected_sets)))
     
    #Calculate scores using bootstrapping
    for i in range(0, repeats):
        print("Iteration %i/%i"%(i,repeats))
        for j in range(0, n_treatment_groups):
            treatment_group_name = None if j == n_treatment_groups -1 else treatment_groups[j]
            bootstrap_scores[j,i,:] = filtered_group_harmonization_assessment('treatment_group',treatment_group_name,'Location',selected_sets, feature_sets, protocol_data, impute=impute, sample=False, scoring=scoring)
    
    harmonization_report = np.zeros((len(selected_sets),9))
    scores = np.zeros((n_treatment_groups,n_selected_sets))
    
    #Calculate the scores
    for j in range(0, n_treatment_groups):
        treatment_group_name = None if j == n_treatment_groups -1 else treatment_groups[j]
        scores[j,:] = filtered_group_harmonization_assessment('treatment_group',treatment_group_name,'Location',selected_sets, feature_sets, protocol_data, impute=impute, sample=False, scoring=scoring)
    #Fill  the harmonization report
    for i in range(0,n_selected_sets):       
        for j in range(0, n_treatment_groups):        
            cil, ciu = confidence_from_bootsrap(bootstrap_scores[j,:,i])
            harmonization_report[i,j*3 + 0] = scores[j,i]
            harmonization_report[i,j*3 +1] = cil        
            harmonization_report[i,j*3 +2] = ciu  
      
    df = pd.DataFrame( data = harmonization_report, columns = report_columns, index=selected_sets) 
    df.to_csv(file_name)

