#Wrangle Zillow
#Imports for functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from env import host, user, password 
import os
from sklearn.model_selection import train_test_split

import sklearn.preprocessing

#Connection function to access Codeup Database and retrieve zillow dataset from mysql
def get_connection(db, user=user, host=host, password=password):
    '''
    This function creates a connection to Codeup Database with 
    info from personal env file (env file has user login information).
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'  

##############ACQUIRE##################

def acquire_zillow():
    '''
    This function reads in the zillow data from the Codeup 
    Database connection made from get_connection
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
                SELECT parcelid, bedroomcnt, bathroomcnt, buildingqualitytypeid, yearbuilt, 
                regionidcounty, regionidzip, fips, latitude, longitude, calculatedfinishedsquarefeet, 
                taxamount, taxvaluedollarcnt
                FROM  properties_2017
                JOIN predictions_2017 USING(parcelid)
                WHERE transactiondate between "2017-05-01" and "2017-08-31"
                and unitcnt = 261 or 263 or 273 or 274 or 276 or 279;
                '''

    
    return pd.read_sql(sql_query, get_connection('zillow'))    


def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and 
    writes data to a csv file if cached == False. If cached == True 
    reads in zillow df from a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = acquire_zillow()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    return df



#####PREPARE#######

#CLEAN

def clean_zillow(df):
    '''
    Takes in a df of zillow data acquired from sql_query
    and cleans the data appropriately by:
    - handling null values by dropping and imputing
    - converting float variables that do not require a decimal to ints
    - dropping outliers
    - changing columns names
    - drop columns
    - adding new columns
    - scaling data
    return: df, a cleaned pandas dataframe
    '''
    #drop buildingqualitytypeid right now until I figure out best impute method
    df.drop(['buildingqualitytypeid'], axis=1, inplace=True)
    #drop nulls
    df = df.dropna()
    #add two new columns
    df['property_age'] = 2021 - df.yearbuilt
    df['tax_rate'] = (df.taxamount / df.taxvaluedollarcnt)
    #change data types from float to int
    df['bedroomcnt'] = df.bedroomcnt.astype('int')
    df['yearbuilt'] = df.yearbuilt.astype('int')
    df['regionidzip'] = df.regionidzip.astype('int')
    df['regionidcounty'] = df.regionidcounty.astype('int')
    df['fips'] = df.fips.astype('int')
    df['latitude'] = df.latitude.astype('int')
    df['longitude'] = df.longitude.astype('int')
    df['property_age'] = df.property_age.astype('int')
    df = df.set_index("parcelid")
    #rename columns
    df = df.rename(columns={"bedroomcnt": "bedrooms", 
                            "bathroomcnt": "bathrooms", 
                            "calculatedfinishedsquarefeet": "square_feet", 
                            "taxvaluedollarcnt": "tax_value",
                            "regionidzip": "zip_code",
                            "regionidcounty": "county"})

    return df

#generic split
def split_data(df):
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=123)
    return train, validate, test

#split with 
def train_validate_test_split(df, target, seed):
    '''
    spilts our data  into train, validate, test
    by taking in a dataframe and dividing into
    separate
    '''
    # Train, Validate, and test
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed)
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed)
    
    # Split with X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test   

#wrangle: acquire and prep data set
def wrangle_zillow():
    '''
    wrangle_zillow will read in our zillow dataset as a pandas df,
    clean the data,
    split the data,
    return: train, validate, test sets of pandas dataframes from zillow data
    
    '''
    df = clean_zillow(acquire_zillow())

    return split_data(df)






#SCALE
def Standard_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs
    """

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled

def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled
