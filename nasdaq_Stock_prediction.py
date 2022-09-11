#!/usr/bin/env python
# coding: utf-8

# In[1]:


#downloading NASDAQ Price Data
#importing packages
get_ipython().system('pip install yfinance')

import yfinance as yf
import pandas as pd

#download NASDAQ daily stocks index prices
nasdaq = yf.Ticker("^IXIC")

#querying the hsitorical prices to query all data from the very beginning when the index was created
nasdaq = nasdaq.history(period="max")
nasdaq


# In[2]:


#looking at the index
nasdaq.index


# In[3]:



#plotting data (closing price against the index)
nasdaq.plot.line(y = "Close", use_index = True)


# In[4]:


#cleaning and visualising data
del nasdaq["Dividends"]
del nasdaq["Stock Splits"]


# In[5]:


#setting target for Machine Learning - will the price go up or down tomorrow?

#setting tomorrow's column using shift function on close column
nasdaq["tomorrow_price"] = nasdaq["Close"].shift(-1)
nasdaq


# In[6]:


#target boolean as an integer
#1 = increase in price, 0 = decrease in price
nasdaq["target"] = (nasdaq["tomorrow_price"] > nasdaq["Close"]).astype(int)
nasdaq


# In[7]:


#removing data before a certain period
nasdaq = nasdaq.loc["1990-01-01":].copy()
nasdaq


# In[8]:


#Training machine learning model 

from sklearn.ensemble import RandomForestClassifier

#n_estimators is the number of trees I want to trade,
#min_sample_splits is the set low in order to avoid  overfitting

model = RandomForestClassifier(n_estimators=500, min_samples_split=100, random_state=1) 

#-splitting dataset into time and train set 
#-- all rows except last 100 rows in train set
train = nasdaq.iloc[:-100]
#-- last 100 rows in test set
test = nasdaq.iloc[-100:]


#---creating a list called predictors and using it to predict traget for the training set
predictors =["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["target"]) 


# In[9]:


from sklearn.metrics import precision_score

#using precision score accuracy metric (precentage) 
#to check whether the price actually went up when we predicted it'd go up

preds = model.predict(test[predictors])
preds


# In[10]:


#-turning numpy array into pandas series
preds = pd.Series(preds, index=test.index, name="Predictions")

#-testing precision score 
precision_score(test["target"], preds)


# In[11]:


#-plotting predictions by combining tested values and predicted values
combined = pd.concat([test["target"], preds], axis=1)
combined.plot()


# In[12]:


#Builidng a backtesting system

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["target"], preds], axis=1)
    return combined


# In[13]:


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = [] #list of df with predicitions of single year
    
    for i in range(start, data.shape[0], step):
        #train set is all the year prior to current year
        train = data.iloc[0:i].copy()
        #test set is current year
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[14]:


predictions = backtest(nasdaq, model, predictors)


# In[15]:


#predictions of days market goes up vs down
predictions["Predictions"].value_counts() 


# In[16]:


precision_score(predictions["target"], predictions["Predictions"])


# In[17]:


#to check whehther the precision score is good or not
#looking the precentage of days the market actually went up (as a bench mark)
predictions["target"].value_counts() /  predictions.shape[0]


# In[18]:


#Adding more predictos of improve the accuracy of the algorithm

#creating rolling averages (2 days, trading weeks, 3 months, year, 5 years)
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = nasdaq.rolling(horizon).mean()
    
    #new columns
    ratio_column = f"close_ratio_{horizon}"
    nasdaq[ratio_column] = nasdaq["Close"] / rolling_averages["Close"]
    
    trend_column = f"trend_{horizon}"
    nasdaq[trend_column]= nasdaq.shift(1).rolling(horizon).sum()["target"]
    
    new_predictors += [ratio_column, trend_column]
    


# In[19]:


nasdaq.tail(10)


# In[20]:


#improving model
model = RandomForestClassifier(n_estimators=700, min_samples_split=50, random_state=1) 


# In[21]:


#improving predictor function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train['target'])
    preds = model.predict_proba(test[predictors]) [:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["target"], preds], axis=1)
    return combined


# In[22]:


nasdaq = nasdaq.dropna()
predictions = backtest(nasdaq, model, new_predictors)


# In[23]:


predictions["Predictions"].value_counts()


# In[24]:


precision_score(predictions["target"], predictions["Predictions"])

