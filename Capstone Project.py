#!/usr/bin/env python
# coding: utf-8

# ## importing Neccesaary libraries

# In[1]:


#Data Manipulating & Handling

import pandas as pd
import numpy as np

#Data Visualization libiraies

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci

## Data Preprocessing &  EDA libraris
from collections import OrderedDict

#warning filter libiraries

import warnings
warnings.filterwarnings("ignore")


# ## loading the Dataframe

# In[2]:


df = pd.read_excel("Capstone Project.xlsx")


# In[3]:


df.head()


# ## problem statement - To predict strength of the cement by studying 8 different features

# ## Exploratory Data ananlysis EDA

# In[4]:


df.info()


# Observations from df.info()
# 1. No Null Values
# 2. 1030 Records and 9 features 
# 3. Except age all the features are datatype  is float
# 4. Dependent variable is strenght and rest are independent variable 
# 

# In[5]:


df.describe()


# Analysis from Descriptive Statistics :
# 1. Thier might be skewness in the columns
# 2. Thier might be chance of outliers if we compare quratiles of few columns ( age, cement, slag, superplastic, strength in the upper whisker region )
# 3. Since min & Q1 value is same for slag & ash , we can say that we dont have outliers in the lower whisker region.
# 4. Range values in age is from 1 to 365 , values are in days

# In[6]:


def Custom_Summary(my_df):
    result = []
    for col in my_df.columns:
        if my_df[col].dtypes != 'objects':
            stats = OrderedDict({
                "Feature Name" : col,
                "Count" : my_df[col].count(),
                "Minimum" : my_df[col].min(),
                "Quartile1" : my_df[col].quantile(0.25),
                "Quartile2" : my_df[col].quantile(0.5),
                "Quartile3" : my_df[col].quantile(0.75),
                "mean" : my_df[col].mean(),
                "variance" : round(my_df[col].var()),
                "standard Deviation" : my_df[col].std(),
                "skewness" : my_df[col].skew(),
                "Kurtosis" : my_df[col].kurt()
                
            })
            result.append(stats)
            
    result_df = pd.DataFrame(result)
    
    #Skewness type :
    
    skewness_label = []
    
    for i in result_df["skewness"]:
        if i <= -1 :
            skewness_label.append("Highly negatively skewed")
        elif -1 < i <= -0.5:
             skewness_label.append("Moderately negatively skewed")
        elif -0.5 < i <= 0:
             skewness_label.append("Fairly negatively skewed")
        elif 0 <= i < 0.5:
             skewness_label.append("Fairly positively skewed")
        elif 0.5 <=i <1:
             skewness_label.append("Moderately positively skewed")
        elif i >= 1:
             skewness_label.append("Highly positively skewed")
                
    result_df["skewness comment"] = skewness_label
    
    Kurtosis_label = []
        
    for i in result_df["Kurtosis"]:
        if i >= 1 :
            Kurtosis_label.append("Lepotokurtic curve")
        elif i <= -1:
             Kurtosis_label.append("platykurtic curve")
        else:
             Kurtosis_label.append("Mesokurtic skewed")
                            
    result_df["Kurtosis comment"] = Kurtosis_label
    
    outliers_label = []
    
    for col in my_df.columns:
        if my_df[col].dtypes != "object":
            q1 = my_df[col].quantile(0.25)
            q2 = my_df[col].quantile(0.5)
            q3 = my_df[col].quantile(0.75)
            IQR = q3 - q1
            LW = q1 - 1.5 * IQR
            UW = q3 + 1.5 * IQR
            
            if len(my_df[(my_df[col]< LW) | (my_df[col]> UW)]) > 0 :
                outliers_label.append("Have outliers")
            else:
                outliers_label.append("No outliers")
                
    result_df["outlier comment"] = outliers_label
    
    return result_df   


# In[7]:


Custom_Summary(df)


# # Analysis from Costum summary 

# 1.Cement has misokurtic Arrow which implies the data points are moderate in distance from the mean, So mean and Std are moderate  
# 2.Slag meso kurtic curve
# 3.Ash has playtikurtic Curve which implies the mean doesnt represent whole data properly so Standard deviation is high
# 4.Superplastic has a liptokurtic curve which implies datapoints are closer to the mean
# 
# 

# In[8]:


def replace_outlier(my_df,col,method = 'Quartile', strategy = 'median'):
    col_data = my_df[col] 
    
    if method =='Quartile':
        #Using quartile to calculate IQR 
        q1 = col_data.quantile(0.25)
        q2 = col_data.quantile(0.50)
        q3 = col_data.quantile(0.75)
        
        IQR = q3 -q1
        LW = q1 - 1.5*IQR
        UW = q3 + 1.5*IQR
        
    elif method == 'Standard Deviation':
        # Using SD method
        mean = col_data.mean()
        std = col_data.std()
        LW = mean -2*std
        UW = mean +2*std
    else:
        print("Pass a correct method")
        
    # printing all the outliers
    
    outliers = my_df.loc[(col_data < LW) | (col_data > UW)]
    outlier_density =round(len(outliers)/len(my_df),2) *100
    
    if len(outliers) == 0:
        print(f'Feature{col} does not have outliers')
        print("\n")
    else:
        print(f'Feature {col} has outliers')
        print("\n")
        print(f'Total no.of outliers in {col} or {len(outliers)}')
        print('\n')
        print(f'outlier percentage in {col} is {outlier_density}%')
        print('\n')
        display(my_df[(col_data < LW ) | (col_data > UW)])
              
              
    #Replacing Outliers
              
    if strategy == 'median':
        my_df.loc[(col_data < LW) | (col_data > UW), col] =q2
    elif strategy == 'mean':
        my_df.loc[(col_data < LW) | (col_data > UW), col] = mean
    else:
        print('Pass the correct strategy')
              
    
    return my_df
       
              
              
        


# In[9]:


replace_outlier(df,'age')


# ## ODT Plots (Outlier detection techniques)

# 1. Descriptive Plots
# 2. Histogram with outliers
# 3. Histogram without outliers

# In[10]:


def odt_plots(my_df, col) :
    
    f,(ax1 , ax2 , ax3) = plt.subplots(1 , 3, figsize = (25, 8))  # 1 row and 3 columns in the output plot
    
    # descriptive statistics Box plot
    
    sns.boxplot(my_df[col] , ax = ax1)
    ax1.set_title(col + "Boxplot")
    ax1.set_xlabel("values")
    ax1.set_ylabel("Boxplot")
    
    #plotting Histogram with outliers
    
    sns.distplot(my_df[col] , ax = ax2 ,fit = sci.norm)
    ax2.axvline(my_df[col].mean(), color = "Green")
    ax2.axvline(my_df[col].median() , color = "brown")
    ax2.set_title(col + "Histogram with outliers")
    ax2.set_ylabel("Density")
    ax2.set_xlabel("Values")
    
    #Replacing outliers 
    
    df_out = replace_outlier(my_df, col)
    
    #plotting histogram without outliers
    
    sns.distplot(df_out[col], ax = ax3 , fit = sci.norm)
    ax3.axvline(df_out[col].mean(), color = "Green")
    ax3.axvline(df_out[col].median() , color = "brown")
    ax3.set_title(col + "Histogram with outliers")
    ax3.set_ylabel("Density")
    ax3.set_xlabel("Values")
    plt.show()


# In[11]:


for col in df.columns:
    odt_plots(df,col)

