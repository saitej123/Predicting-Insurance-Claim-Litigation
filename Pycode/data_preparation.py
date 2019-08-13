### UCSB Project 3: Predicting Litigated Claims
### Author: Aaron Barel, Mingxi Chen, Syen Yang Lu
### Descriptions: Modules used for Data Preparation

# Hide all warnings
import warnings
warnings.filterwarnings('ignore')

# Import basic modules
import os
import numpy as np
import pandas as pd

# feature extraction packages from sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize # Scale Normalization

def read_data(input_source='data/clean-data/'):
    '''Define a function that reads in text and quan data
    Input: input source (optional)
    Output: quan and text dataframes'''
    
    # Read in text only and quan only dataset
    quan_only = pd.read_csv(os.path.join(input_source,'quan_only.csv'))
    text_only = pd.read_csv(os.path.join(input_source,'text_only.csv'))
    
    return (quan_only, text_only)
    
def data_prep(minTF=10, cap=20):
    '''Define a function that prepares the data for analysis
    Input: minumum text frequency (optional), claimant cap (optional)
    Output: Prepared Dataframe and column names dictionary'''
    
    # Read in quan only and text only data
    quan_only, text_only = read_data()
    
    # Create dummy variables for categorical data
    quan_only = create_dummies(quan_only)
    
    # Create term docuement matrix for text data
    text_only = termDocumentMatrix(text_only, minTF)

    # Combine quan only and text only dataframe
    key = ['UNIT_NUMBER', 'CLAIM_NUMBER', 'RANDOM_NUMBER', 'LITIGATION']
    label = ['TARGET']
    var_to_join = key + label
    combined = quan_only.merge(text_only, on=var_to_join)

    combined = down_sample(combined,'CLAIM_NUMBER','UNIT_NUMBER', cap) # Shrink the number of claimants down to at most 20 claimants 
    combined = scale_numeric(combined) # Scale continuous variables
    
    # Define a dictionary to records the column names
    colDict = {}
    colDict['keyCol'] = key
    colDict['labelCol'] = label
    colDict['quanCol'] = quan_only.columns.drop(var_to_join).tolist()
    colDict['textCol'] = text_only.columns.drop(var_to_join).tolist()
    
    # Reorder the columns of final dataset
    combined = combined[colDict['keyCol']+colDict['labelCol']+colDict['quanCol']+colDict['textCol']]
    return (combined, colDict)

def create_dummies(data):
    '''Defines a function that creates dummy variables for each categorical variable
    Input: A dataframe containing categorical columns
    Output: A dataframe with one hot encoders for each categorical column'''
    
    # Extract a list of categorical columns
    category = ['INSURANCE_GROUP_IND','CAUSE_OF_LOSS','CLAIM_LOSS_TYPE', \
                'CLAIMANT_GENDER','POLICY_DESCRIPTION_GROUPED','IS_SAME_STATE']

    # Create dummy variables for each category
    cat_df = pd.get_dummies(data[category])
    other_df = data.drop(columns=category)
    
    # Combine the result and return it
    all_column = [other_df, cat_df]
    return pd.concat(all_column, axis=1)

def termDocumentMatrix(data, minDF=1):
    ''' Define a function that creates term document matrix from text data
    Input: A dataframe containing text field
    Output: A term-document matrix'''

    # Extract the text column from data
    text = data['TEXT']
    
    # Fit text data into count vectorizer
    count_vec = CountVectorizer(min_df=minDF)
    count_vec.fit(text)
    
    # Extract words and corresponding counts in each claim
    tdm = count_vec.transform(text).toarray()
    words = count_vec.get_feature_names()
    
    # Convert term document matrix into dataframe
    text_df = pd.DataFrame(data=tdm, columns=words)
    other_df = data.drop(columns='TEXT')
    
    # Combine the results and return it
    all_df = [other_df, text_df]
    return pd.concat(all_df, axis=1)

def down_sample(df, indexOfGroup, indexOfMember, claim_sample):
    '''Undersample claims that have more than 20 claimants
    Input:
    df - dataframe
    indexOfGroup - column at which we wish to sample from (CLAIM_NUMBER)
    indexOfMember - column at which we distinguis the members of the group (UNIT_NUMBER)
    claim_sample (int) - sample size for claims that exceed this count
    
    Output: Dataframe where all claims have no more than certain number of claimants'''
    
    # Count the number of claimants for each claim and sort them by descending order
    claim_count = df[indexOfGroup].value_counts()
    claim_count = claim_count.sort_values(ascending = False)

    # Extract claims that exceed a certain threshold
    sampleIndicies = claim_count[claim_count > claim_sample].index.tolist()
    subset = df.loc[df[indexOfGroup].isin(sampleIndicies)] # Subset the entire dataset
    totalMembers = subset[indexOfMember].values # Extract unit number
    
    # Undersample claims that exceed the threshold down to the threshold level
    keep = np.array([])
    for ind in sampleIndicies:
        subsets = df.loc[df[indexOfGroup] == ind]
        units = subsets[indexOfMember].values
        k = np.random.choice(units, claim_sample, replace = False)
        keep = np.append(keep, k)

    # Compute the list of unit number that will be removed
    removeIndicies = np.setdiff1d(totalMembers, keep)

    # Remove the unwanted unit number from the dataframe
    sampled_df = df.loc[-df[indexOfMember].isin(removeIndicies)]
    sampled_df.index = range(len(sampled_df[indexOfMember]))
    
    return sampled_df

def scale_numeric(data):
    '''Defines a function that scale all the continuous features
    Input: A dataframe with continuous variables
    Output: A dataframe with normalized variables'''
    
    # Get a list of continous variables
    quantitative = ['CLAIMANT_AGE','REPORT_LAG','UNIT_CREATION_LAG']
    
    # Normalize and convert the result into dataframe
    quan_df = pd.DataFrame(normalize(data[quantitative]), columns=quantitative)
    other_df = data.drop(columns=quantitative)
    
    # Combine the scaled features and other columns and return the resulting dataframe
    all_column = [quan_df, other_df]
    return pd.concat(all_column, axis=1)

def train_test_split(data, holdout, setSeed=0):
    '''Defines a function that split data into training and test sets
    Input: A dataframe and the proportion of data as holdout set
    Output: Test and training data'''
    
    key = 'RANDOM_NUMBER'
    label = ['TARGET']
    
    # Gets Unique Claim Numbers
    claimNo = data[key].unique()

    # Selects part of Claim Numbers for test set
    np.random.seed(setSeed)
    testNo = np.random.choice(claimNo, int(holdout*len(claimNo)), replace = False)

    # Split data into test and training sets 
    testSet = data.loc[data[key].isin(testNo)]
    trainSet = data.loc[-data[key].isin(testNo)]
    
    return (trainSet, testSet)

def define_fold(data):
    ''' Set fold definition for the data by random number
    Input: A dataframe with unit number
    Output: A dataframe with created random number'''
    
    # Divide random number into 5 folds
    sample = data['RANDOM_NUMBER'].unique()
    np.random.shuffle(sample)
    folddef = np.array_split(sample,5)
    
    # Assign partition number based on the fold
    def assign_partition(x, fold):
        if x in fold[0]:
            return(0)
        elif x in fold[1]:
            return(1)
        elif x in fold[2]:
            return(2)
        elif x in fold[3]:
            return(3)
        elif x in fold[4]:
            return(4)
        else:
            return(None)
    
    return data['RANDOM_NUMBER'].apply(lambda x: assign_partition(x, folddef))