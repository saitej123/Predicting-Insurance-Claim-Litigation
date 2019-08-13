### UCSB Project 3: Predicting Litigated Claims
### Author: Aaron Barel, Mingxi Chen, Syen Yang Lu
### Descriptions: Modules used for Data Analysis

# Hide all warnings
import warnings
warnings.filterwarnings('ignore')

# Import basic modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# feature extraction packages from sklearn
from sklearn.feature_extraction.text import TfidfTransformer

# Import machine learning metrics from sklearn and statmodels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Import packages for sampling techniques
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Import packages for model selection
from sklearn.model_selection import ParameterGrid

# Import packages for classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

# Set the visual for pandas dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def compare_sampling(model_name, train_df, test_df, col_dict):
    '''Compare result of different sampling methods
    Input: Name of model, train data, test data, column dictionary
    Output:  Dataframe containing summaries of all results'''
    
    # Get model for balanced and unbalanced method
    unbalanced_model = get_model(model_name, {})
    balanced_model = get_model(model_name, {'class_weight': 'balanced'})
    
    # Train model with all different methods
    baseline_result = model_result(unbalanced_model, train_df, test_df, col_dict, method=None)
    undersampling_result = model_result(unbalanced_model, train_df, test_df, col_dict, method='RandomUnderSampler')
    oversampling_result = model_result(unbalanced_model, train_df, test_df, col_dict, method='RandomOverSampler')
    smote_result = model_result(unbalanced_model, train_df, test_df, col_dict, method='SMOTE')
    balanced_result = model_result(balanced_model, train_df, test_df, col_dict, method=None)

    # Combine the result
    compare_result = pd.DataFrame([baseline_result, undersampling_result, oversampling_result, smote_result, balanced_result])
    compare_result.index=['None', 'Rus', 'Ros', 'Smote', 'Balanced']
    return compare_result

def get_model(model_name, param):
    '''Create an instance of model using model name and parameters
    Input: Name of the model and parameters
    Output: Model object based on model name'''
    
    # Initialize the model with the given parameters
    if model_name == 'support_vector_machine':
        model = SVC(**param)
    elif model_name == 'gaussian_naive_bayes':
        model = GaussianNB(**param)
    elif model_name == 'complement_naive_bayes':
        model = ComplementNB(**param)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(**param)
    elif model_name == 'elasticnet':
        model = SGDClassifier(**param)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(**param)
    else:
        model = None
        
    return model

def model_result(model, train_df, test_df, col_dict, method=None):
    '''Perform prediction on test data and summarize model result
    Input: Model object, train data, test data, column dictionary, method (optional)
    Output: Dataframe summarizing model results'''
    
    # Compute evaluation metrics
    Xtr, Xvl, Ytr, Yvl, predYtr, predYvl = model_evaluation(model, train_df, test_df, col_dict)
    
    result = {}
    
    result['train_accuracy'] = accuracy_score(Ytr, predYtr)
    result['test_accuracy'] = accuracy_score(Yvl, predYvl)
    
    result['train_precision'] = precision_score(Ytr, predYtr)
    result['test_precision'] = precision_score(Yvl, predYvl)
    
    result['train_recall'] = recall_score(Ytr, predYtr)
    result['test_recall'] = recall_score(Yvl, predYvl)
    
    result['train_f1'] = f1_score(Ytr, predYtr)
    result['test_f1'] = f1_score(Yvl, predYvl)
    
    result['roc_auc'] = roc_auc_score(Yvl, predYvl)
    
    return result

def model_evaluation(model, train_df, test_df, col_dict, method=None):
    ''' Get a set of train and test features and target as well as prediction
    Input: Model object, train data, test data, column dictionary, method (optional)
    Output: Train feature and target, test feature and target, train and test predicted target'''
    
    # Extract list of column names
    keyCol = col_dict['keyCol']
    labelCol = col_dict['labelCol']
    quanCol = col_dict['quanCol']
    textCol = col_dict['textCol']
    
    # Get a list of train index and perform resampling on the data
    train_index = train_df[keyCol]
    train_df = resampling(train_df.drop(columns=keyCol), method=None)

    # Set train feature and target
    Ytr = train_df[labelCol]
    Xtr = train_df.drop(columns=labelCol)
    
    # Set test feature and target
    Yvl = test_df[labelCol]
    Xvl = test_df.drop(columns=keyCol+labelCol)
        
    # Perform TFIDF on the text column
    try:
        Xtr = TFIDF(Xtr, textCol)
    except:
        pass
    
    # Fit the modelo
    model.fit(Xtr, Ytr)    
    
    # Make prediction on the train and test data
    predYtr = model.predict(Xtr)
    predYvl = model.predict(Xvl)
    
    # Return all moving parts
    return (Xtr, Xvl, Ytr, Yvl, predYtr, predYvl)

def resampling(data, method=None):
    '''The function returns data based on sampling method
    Input: Dataframe, method (optional)
    Output: Resampled dataframe based on method'''
    
    # Return coresponding result based on resampling method
    if method == 'RandomUnderSampler':
        return undersampling(data)
    elif method == 'RandomOverSampler':
        return oversampling(data)
    elif method == 'SMOTE':
        return smote_sampling(data)
    else:
        return data

def undersampling(data):
    '''Define a function that perform sampling using random under sampler
    # Input: A data frame without label, a list of corresponding label
    # Output: A balanced data frame and label'''

    label = ['TARGET']

    # Divide data into feature and target
    y = data[label]
    X = data.drop(columns=label)
    
    # Extract column names of feature and target
    y_labels = y.columns
    X_labels = X.columns
    
    # Perform resampling
    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    # Set resampled data into dataframe
    y_df = pd.DataFrame(y_resampled, columns=y_labels)
    X_df = pd.DataFrame(X_resampled, columns=X_labels)
    
    # Combined resampled feature and target and return the result
    return pd.concat([X_df, y_df], axis=1)

def oversampling(data):
    '''Define a function that perform sampling using random over sampler
    # Input: A data frame without label, a list of corresponding label
    # Output: A balanced data frame and label'''

    label = ['TARGET']
    
    # Divide data into feature and target
    y = data[label]
    X = data.drop(columns=label)
    
    # Extract column names of feature and target
    y_labels = y.columns
    X_labels = X.columns
    
    # Perform resampling
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Set resampled data into dataframe
    y_df = pd.DataFrame(y_resampled, columns=y_labels)
    X_df = pd.DataFrame(X_resampled, columns=X_labels)
    
    # Combined resampled feature and target and return the result
    return pd.concat([X_df, y_df], axis=1)

def smote_sampling(data):
    '''Define a function that perform sampling using SMOTE
    # Input: A data frame without label, a list of corresponding label
    # Output: A balanced data frame and label'''

    label = ['TARGET']
    
    # Divide data into feature and target
    y = data[label]
    X = data.drop(columns=label)
    
    # Extract column names of feature and target
    y_labels = y.columns
    X_labels = X.columns
    
    # Perform resampling
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Set resampled data into dataframe
    y_df = pd.DataFrame(y_resampled, columns=y_labels)
    X_df = pd.DataFrame(X_resampled, columns=X_labels)
    
    # Combined resampled feature and target and return the result
    return pd.concat([X_df, y_df], axis=1)
    
def TFIDF(data, textCol):
    '''Defines a function to weigh term document matrix using TF-IDF
    Input: Term Document Matrix
    Output: TF-IDF weighted Term Document Matrix'''

    # Get all the text columns
    termDocMat = data[textCol]
    
    # Tranforms text columns by assigning TFIDF weights
    tfidf_vec = TfidfTransformer()
    tfidf_vec.fit(termDocMat)
    tfidf_mat = tfidf_vec.transform(termDocMat).toarray()
    
    # Return the tfidf weighted term document matrix
    return(pd.DataFrame(data=tfidf_mat, index=data.index, columns=data.columns))

def model_cv(model_name, param_list, data, folddef, col_dict):
    '''Defines a function to perform cross validation for each set of parameters
    Input: Name of model, Set of parameter, data, fold definition and column dictionary
    Output: Dataframe summarizing all result metrics'''
    
    cv_result = []

    # Perform cross validation over different sets of paramter 
    for param in param_list:
        # Calculate cv score by taking the mean of each iteration
        cv_score = np.mean([cross_validation(model_name, param, x, folddef, data, col_dict) for x in range(0,5)],axis=0)
        cv_result.append(cv_score)
    
    # Define a set of metrics
    metrics = ['train_accuracy',
               'test_accuracy',
               'train_precision',
               'test_precision',
               'train_recall',
               'test_recall',
               'train_f1',
               'test_f1',
               'roc_auc']
    
    # Combine the result and return as a dataframe
    paramList = pd.DataFrame(param_list)
    resultList = pd.DataFrame(cv_result, columns=metrics)
    
    all_df = [paramList, resultList]
    return pd.concat(all_df, axis=1)

def cross_validation(model_name, param, chunkid, chunkdef, data, col_dict):
    '''Defines a function that sets data aside for cross validation
    Input: Name of model, set of parameters, chunk id, chunk definition, data, column dictionary
    Output: Calculated metrics for that iteration'''
    
    # Define test and train index
    test = list(chunkdef == chunkid)
    train = list(chunkdef != chunkid)
    
    # Separate data into test and train data
    trainDF = data[train]
    testDF = data[test]
    
    # Get model and its result
    model = get_model(model_name, param)
    result = model_result(model, trainDF, testDF, col_dict)
    
    return tuple(result.values())

def reg_summary(model, col_dict):
    '''Defines a function to compute regression summary
    Input: model, column dictionary
    Output: DAtaframe containing regression Summary'''
    
    # Extract the list of all features
    features = col_dict['quanCol'] + col_dict['textCol']
    
    # Define a dictinary to summary variable and coefficientss
    lr_dict = {}
    lr_dict['variable'] = ['intercept'] + features
    lr_dict['coefficient'] = list(model.intercept_) + list(model.coef_[0])
    
    # Return the resulting dataframe
    return pd.DataFrame(lr_dict)
    
def var_imp(model, train_df, test_df, col_dict):
    '''Defines a function that computes variable importance
    Input: Model object, train data, test data, column dictionary
    Output: Dataframe with column names and importance scores'''
    
    # Compute the performance
    Xtr, Xvl, Ytr, Yvl, predYtr, predYvl = model_evaluation(model, train_df, test_df, col_dict)
    
    # Extract column names to drop
    keyCol = col_dict['keyCol']
    labelCol = col_dict['labelCol']
    
    # Get the list of column names and importance score
    var = train_df.drop(columns=keyCol+labelCol).columns
    varImp = model.feature_importances_
    
    # Return the summaryof variable importance in a dataframe
    return pd.DataFrame({'var':var,'varImp':varImp}).sort_values('varImp',ascending=False)

def var_imp_plot(var_imp_result, num_var=10):
    '''Defines a function that plot variable importance
    Input: Dataframe containing summary of variable importance, number of variables to show on the graph (optional)
    Output: Variable Importance Plot'''
    
    # Sort the value of variable importance for the top few variables
    var_imp_result = var_imp_result[:num_var].sort_values('varImp')
    
    # Plot the result
    plt.figure(dpi=100)
    plt.barh(var_imp_result['var'], var_imp_result['varImp'])
    plt.xlabel('Importance Score')
    plt.title('Variable Importance')
    plt.show()

def plot_lift_chart(model, train_df, test_df, col_dict):
    '''Defines a function that computes lift chart
    Input: Model object, train data, test data, column dictionary
    Output: Dataframe of lift chart'''
    
    # Calculate the probability for the model
    Ytr, Yvl, train_proba, test_proba = get_proba(model, train_df, test_df, col_dict)
    
    # Get the actual target from test and train datta
    train_actual = train_df['TARGET']
    test_actual = test_df['TARGET']
    
    # Record the predicted probability and actual target in a dictionary
    test_dict = {}
    test_dict['test_proba'] = test_proba
    test_dict['test_actual'] = test_actual
    
    # Divide the proabilities into 10 quantiles
    test_result = pd.DataFrame(test_dict) \
                .sort_values('test_proba') \
                .reset_index()
    test_result['range'] = pd.cut(test_result.index, 10)
    
    # Set value of upper and lower bound probability
    lower = test_result.groupby('range')['test_proba'].min().tolist()
    upper = test_result.groupby('range')['test_proba'].max().tolist()
    upper[-1] = 1
    
    # Calculate the metric needed in the lift chart
    lift_chart = test_result.groupby('range').sum()
    lift_chart['amount'] = test_result.groupby('range')['index'].count().tolist()
    lift_chart['quantile'] = lift_chart.reset_index().index + 1
    lift_chart['lower'] = lower
    lift_chart['upper'] = upper
    
    # Clean up the dataframe
    selectCol = ['quantile', 'amount', 'lower', 'upper', 'test_proba', 'test_actual']
    lift_chart = lift_chart.reset_index()[selectCol] \
                            .round({'lower': 3, 'upper': 3, 'test_proba':1}) \
                            .rename(columns={'quantile':'Quantile', 'amount':'Amount', \
                                             'lower':'Lower Bound', 'upper':'Upper Bound',
                                             'test_proba': 'Predicted #', 'test_actual':'Actual #'})
    
    return lift_chart

def get_proba(model, train_df, test_df, col_dict):
    '''Defines a function that caluclate prediction probabilties
    Input: Model object, train data, test data, column dictionary
    Output: Train and test, actual and predicted target'''
    
    # Extract column names
    keyCol = col_dict['keyCol']
    labelCol = col_dict['labelCol']
    quanCol = col_dict['quanCol']
    textCol = col_dict['textCol']
    
    # Set the train index
    train_index = train_df[keyCol]
    # Resample the train data based on the method
    train_df = resampling(train_df.drop(columns=keyCol), method=None)

    # Set train feature and target
    Ytr = train_df[labelCol]
    Xtr = train_df.drop(columns=labelCol)
    
    # Set test feature and target
    Yvl = test_df[labelCol]
    Xvl = test_df.drop(columns=keyCol+labelCol)
    
    # Perform TFIDF on text data
    try:
        Xtr = TFIDF(Xtr, textCol)
    except:
        pass
    
    # Fit the model
    model.fit(Xtr, Ytr)    
    
    # Make prediction on the probabilities
    predYtr = model.predict_proba(Xtr)[:,1]
    predYvl = model.predict_proba(Xvl)[:,1]
    
    return (Ytr, Yvl, predYtr, predYvl)

def get_tpr_fpr(model, train_df, test_df, col_dict):
    '''Defines a function that computes fpr and tpr
    Input: Model object, train data, test data, column dictionary
    Output: TPR, FPR'''
    
    # Retreieve a list of train and test, actual and predicted target
    Ytr, Yvl, predYtr, predYvl = get_proba(model, train_df, test_df, col_dict)
    
    # Compute tpr and fpr and return the result
    fpr, tpr, thresholds = roc_curve(Yvl, predYvl)
    return (fpr, tpr)