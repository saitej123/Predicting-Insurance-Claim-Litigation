### UCSB Project 3: Predicting Litigated Claims
### Author: Aaron Barel, Mingxi Chen, Syen Yang Lu
### Descriptions: Modules used for Data Preprocessing

# Hide all warnings
import warnings
warnings.filterwarnings('ignore')

# Import basic modules
import os
import pandas as pd
from hashlib import md5 # Hash Function

def read_data(data_path):
    '''Read in dataset for the purpose of this project
    Input: A path to the folder containing the data files
    Output: Three dataframes for claim_unit, injuries, claim_log_activity'''
    
    claim_unit = pd.read_csv(os.path.join(data_path, 'units_for_UCSB.csv'))
    injuries = pd.read_csv(os.path.join(data_path, 'injuries.csv'))
    claim_log_activity = pd.read_csv(os.path.join(data_path, 'claimlogactivity.csv'))
    
    return (claim_unit, injuries, claim_log_activity)

def load_data(data_path):
    '''Load all datasets and merge them
    Input: None
    Output: Merged dataframe for all data files'''
    
    # Load data into the system
    claim_unit, injuries, claim_log_activity = read_data(data_path)
    
    # Preprocess injuries file by combining multiple injuries for each unit
    injuries_df = injuries.groupby('UNIT_NUMBER', as_index=False) \
                        .agg(lambda x: ' '.join(x.astype(str))) \
                        .drop(['AFFECTED_AREA', 'INJURY_TYPE', 'FATALITY'], axis=1)

    # Preprocess claim unit file by removing unwanted columns
    claim_unit_df = claim_unit.drop(['POLICY_TYPE', 'LOSS_ZIP', 'CLAIMANT_ZIPCODE'], axis=1)

    return pd.merge(injuries_df, claim_unit_df, how='left', on='UNIT_NUMBER')

def data_preprocessing(data):
    '''Defines a function that performs data preprocessing
    Input: A raw dataframe
    Output: A preprocessed dataframe'''
    
    data = var_preprocessing(data) # Preprocessing and grouping of variables, creation of new features
    data = random_number(data) # Generate unique random number identifier
    data = create_new_var(data) # Create new variable based on loss description

    key_col = ['UNIT_NUMBER','CLAIM_NUMBER','RANDOM_NUMBER']
    label_col = ['TARGET','LITIGATION']
    text_col = ['LOSS_DESCRIPTION','INJURY_DESCRIPTION']
    quan_col = ['CLAIMANT_AGE','REPORT_LAG','UNIT_CREATION_LAG']
    cat_col = ['INSURANCE_GROUP_IND','CAUSE_OF_LOSS','CLAIM_LOSS_TYPE','CLAIMANT_GENDER','POLICY_DESCRIPTION_GROUPED','IS_SAME_STATE']
    
    col_to_keep = key_col + label_col + text_col + quan_col + cat_col
    
    return data[col_to_keep]

def var_preprocessing(data):
    '''Defines a function that preprocess the variables, group categories and create new features
    Input: A dataframe
    Output: A dataframe with preprocessed variables'''
    
    # Group policy description to reduce the number of unique values
    data['POLICY_DESCRIPTION_GROUPED'] = data['POLICY_DESCRIPTION'].astype(str) \
                                                                    .apply(lambda x: x[0:4]) \
                                                                    .replace(['GL A', 'WHLS'],['ARTI', 'WHOL']) \
                                                                    .fillna('NONE')
    
    # Convert date into datetime format
    dateVar = ['UNIT_CREATED_DATE','CLAIM_REPORTED_DATE','LOSS_DATE']
    data[dateVar] = data[dateVar].apply(pd.to_datetime)
    
    # Create report lag (time difference between loss date and claim reported date)
    data['REPORT_LAG'] = data['CLAIM_REPORTED_DATE'] - data['LOSS_DATE']
    data['REPORT_LAG'] = data['REPORT_LAG'].apply(lambda x: x.total_seconds()) \
                                            .apply(lambda y: y/(24*3600))
    
    # Create report lag (time difference between claim reported date and unit creation date)
    data['UNIT_CREATION_LAG'] = data['UNIT_CREATED_DATE'] - data['CLAIM_REPORTED_DATE']
    data['UNIT_CREATION_LAG'] = data['UNIT_CREATION_LAG'].apply(lambda x: x.total_seconds()) \
                                                            .apply(lambda y: y/(24*3600))
    
    # Set all negative numbers to 0 
    data.loc[data['CLAIMANT_AGE'] < 0, 'CLAIMANT_AGE'] = 0
    data.loc[data['REPORT_LAG'] < 0, 'REPORT_LAG'] = 0
    data.loc[data['UNIT_CREATION_LAG'] < 0, 'UNIT_CREATION_LAG'] = 0
    
    # Create a variable that identifies difference between loss state and policy state
    data["IS_SAME_STATE"] = data["POLICY_STATE"]==data["LOSS_STATE"]
    
    # Fill in missing value in claimant gender as U 'Unknown'
    data['CLAIMANT_GENDER'] = data['CLAIMANT_GENDER'].fillna('U')
    
    # Fill in missing value in claimant age as 0
    data['CLAIMANT_AGE'] = data['CLAIMANT_AGE'].fillna(0)
    
    return data

def random_number(data):
    '''Generate unique random number for each unit number
    Input: A dataframe containing all unit numbers
    Output: A dataframe with an extra column for random number'''
    
    def md5_hash(unit_number, add_str):
        return(int(md5((unit_number[:10] + add_str).encode('UTF-8')).hexdigest(), 16) / 16.0**32)

    # Apply md5_hash function for each unit number
    data['RANDOM_NUMBER'] = data['UNIT_NUMBER'].apply(lambda x: md5_hash(x,'useforpartition?')) \
                                                .tolist()
    
    return data
       
def create_new_var(data):
    '''Define a function that captures litigated but not flagged claims
    Input: A dataframe containing TARGET variable
    Output: A dataframe containing new variable LITIGATION'''
    
    # Initialize the new variable with TARGET
    data['LITIGATION'] = data['TARGET'].copy()
    
    # Filter all unlitigated claimant
    unlitigate = data[data['LITIGATION'] == 0]
    
    # Find out the cases where the claimant unit is mislabeled
    keywords = ' sue|suit|litigation'
    mislabel = data['LOSS_DESCRIPTION'].str.contains(keywords, case=False, regex=True) ### Need updates
    
    # Set all incorrect unit to 2
    data['LITIGATION'][unlitigate[mislabel].index] = 2
    
    return data