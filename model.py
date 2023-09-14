
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols


from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import math

#evaluation funcitons
def plot_residuals(actual, predicted): #target is actual, yhat is predicted
    '''
    plot_residuals will take in actual and prediction series
    and plot the residuals as a scatterplot.
    '''
    
    residual = actual - predicted
    
    plt.scatter(actual, residual)
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title('Baseline Residuals')
    plt.show
    





def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) **2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()

def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    })

def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def baseline_median_errors(actual):
    predicted = actual.median()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }



def better_than_baseline(actual, predicted):
    rmse_baseline = rmse(actual, actual.mean())
    rmse_model = rmse(actual, predicted)
    return rmse_model < rmse_baseline

def model_significance(ols_model):
    return {
        'r^2 -- variance explained': ols_model.rsquared,
        'p-value -- P(data|model == baseline)': ols_model.f_pvalue,
    }    


def select_kbest(x, y, k):
    
    # parameters: f_regression stats test, give me 8 features
    f_selector = SelectKBest(f_regression, k=k)
    
    # find the top 8 X's correlated with y
    f_selector.fit(X_train_scaled, y_train)
    
    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()
    
    f_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    
    return f_feature

def rfe(x, y, k):
    
    lm = LinearRegression()
    
    rfe = RFE(lm, k)
    
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X_train_scaled,y_train)  
    
    mask = rfe.support_
    
    rfe_features = X_train_scaled.loc[:,mask].columns.tolist()
    
    print(str(len(rfe_features)), 'selected features')
    
    return  rfe_features


def mvp_scatter():
        # Create a scatter plot for tax_value
    plt.figure(figsize=(12, 6))
    plt.scatter(mvp.tax_value.value_counts().index, mvp.tax_value.value_counts().values, alpha=0.5)
    plt.title('Scatter Plot of Tax Value Value Counts')
    plt.xlabel('Tax Value')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

    # Create a scatter plot for square_feet
    plt.figure(figsize=(12, 6))
    plt.scatter(mvp.square_feet.value_counts().index, mvp.square_feet.value_counts().values, alpha=0.5)
    plt.title('Scatter Plot of Square Feet Value Counts')
    plt.xlabel('Square Feet')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

    # Create a scatter plot for bathrooms
    plt.figure(figsize=(12, 6))
    plt.scatter(mvp.bathrooms.value_counts().index, mvp.bathrooms.value_counts().values, alpha=0.5)
    plt.title('Scatter Plot of Bathrooms Value Counts')
    plt.xlabel('Bathrooms')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

    # Create a scatter plot for bedrooms
    plt.figure(figsize=(12, 6))
    plt.scatter(mvp.bedrooms.value_counts().index, mvp.bedrooms.value_counts().values, alpha=0.5)
    plt.title('Scatter Plot of Bedrooms Value Counts')
    plt.xlabel('Bedrooms')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    return mvp




import matplotlib.pyplot as plt

def county_taxrate_distribution(df):
    # Los Angeles County
    lac = df[df.fips == "Los Angeles"].tax_rate
    lac_mean = round(lac.mean(), 3)

    plt.figure(figsize=(12, 8))
    plt.title("Los Angeles County Tax Rate Distribution")
    plt.hist(lac, bins=800)
    plt.vlines(lac_mean, 0, 20000, ls='--', color='orange', label="LAC mean tax rate: 1.38%")
    plt.xlabel('Tax Rate')
    plt.ylabel("Number of Properties")
    plt.legend()
    plt.xlim(0.0, 0.1)
    plt.show()  # Display the plot

    # Orange County
    oc = df[df.fips == "Orange"].tax_rate
    oc_mean = round(oc.mean(), 3)

    plt.figure(figsize=(12, 8))
    plt.title("Orange County Tax Rate Distribution")
    plt.hist(oc, bins=800)
    plt.vlines(oc_mean, 0, 1200, ls='--', color='orange', label="OC mean tax rate: 1.21%")
    plt.xlabel('Tax Rate')
    plt.xlim(0.0, 0.1)
    plt.ylabel("Number of Properties")
    plt.legend()
    plt.show()  # Display the plot

    # Ventura County
    vc = df[df.fips == "Ventura"].tax_rate
    vc_mean = round(vc.mean(), 3)

    plt.figure(figsize=(12, 8))
    plt.title("Ventura County Tax Rate Distribution")
    plt.hist(vc, bins=800)
    plt.vlines(vc_mean, 0, 1200, ls='--', color='orange', label="VC mean tax rate: 1.19%")
    plt.xlabel('Tax Rate')
    plt.ylabel('Number of Properties')
    plt.xlim(0.0, 0.1)
    plt.legend()
    plt.show()  # Display the plot


    
    
def mvp_scatter(data):
    # Create a scatter plot for tax_value
    plt.figure(figsize=(12, 6))
    plt.scatter(data.tax_value.value_counts().index, data.tax_value.value_counts().values, alpha=0.5)
    plt.title('Scatter Plot of Tax Value Value Counts')
    plt.xlabel('Tax Value')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

    # Create a scatter plot for square_feet
    plt.figure(figsize=(12, 6))
    plt.scatter(data.square_feet.value_counts().index, data.square_feet.value_counts().values, alpha=0.5)
    plt.title('Scatter Plot of Square Feet Value Counts')
    plt.xlabel('Square Feet')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

    # Create a scatter plot for bathrooms
    plt.figure(figsize=(12, 6))
    plt.scatter(data.bathrooms.value_counts().index, data.bathrooms.value_counts().values, alpha=0.5)
    plt.title('Scatter Plot of Bathrooms Value Counts')
    plt.xlabel('Bathrooms')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

    # Create a scatter plot for bedrooms
    plt.figure(figsize=(12, 6))
    plt.scatter(data.bedrooms.value_counts().index, data.bedrooms.value_counts().values, alpha=0.5)
    plt.title('Scatter Plot of Bedrooms Value Counts')
    plt.xlabel('Bedrooms')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    
    
    
    
    
    
def visualize_univariates():
    plt.figure(figsize=(10, 8))
    for i, col in enumerate(['bedrooms', 'bathrooms', 'square_feet', 'tax_value']):  
        plot_number = i + 1 # i starts at 0, but plot nos should start at 1
        series = df[col]  
        plt.subplot(2,2, plot_number)
        plt.title(col)
        series.hist(bins=20)
    


def mvp_pairplot():
    sns.pairplot(mvp,
                x_vars=["bedrooms", "bathrooms", "square_feet", 'tax_value'],
                y_vars=["bedrooms", "bathrooms", "square_feet", 'tax_value'],
                kind= "reg")
    

        
