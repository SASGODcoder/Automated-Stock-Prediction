#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import splinter
from splinter import Browser
import time
import os

# In[ ]:


def correl(mylist):
    if len(mylist) == 1:
        print ('Need 2 stocks to correlate')
        return None
    correl_list = []
    for i in range(len(mylist)):
        if i == len(mylist) - 1:
            break
        j = i + 1
        flag = True
        while flag:
            df1 = pd.read_csv(filepath + mylist[i] + ".csv")
            df2 = pd.read_csv(filepath + mylist[j] + ".csv")
            list1 = df1['Close']
            list2 = df2['Close']

            numerator = 0
            d1 = 0
            d2 = 0
            list1_mean = np.mean(list1)
            list2_mean = np.mean(list2)
            for k in range(len(list1)):
                numerator += (list1[k] - list1_mean)*(list2[k] - list2_mean)
                d1 += (list1[k] - list1_mean)**2
                d2 += (list2[k] - list2_mean)**2
            stock_correl = numerator/np.sqrt(d1*d2)
            print (mylist[i] + ' and ' + mylist[j] + ' are correlated by ' + str(stock_correl))
            j += 1
            if j == len(mylist):
                flag = False
            correl_list.append(stock_correl)
    return correl_list


# In[ ]:


ticker = [str(x).upper() for x in input('What stock do you want me to predict? ').split()]
flag = True

while flag:
    question = input("Anything else? 'YES'/'NO' ").upper()
    if question == 'no'.upper():
        flag = False

    if question == 'yes'.upper():
        another_stock = [str(x).upper() for x in input().split()]
        ticker.extend(another_stock)    

days = input("How many days do you want the prediction for? ")
print ("Do you want the start date for training the model?")
start_date = input("If yes, then give it in the form mm/dd/yyyy. Otherwise, type 'NO' ").upper()
if start_date == 'NO':
    start_date = '1/1/2010'
username = input("What's your username? ")
filepath = 'C:\\Users\\' + username + '\\Downloads\\'


# In[ ]:


ticker = np.array(ticker).flatten()
print (ticker)
for i in range(len(ticker)):
    html = urlopen("https://finance.yahoo.com/quote/" + ticker[i] + "/key-statistics?p=" + ticker[i])
    soup = BeautifulSoup(html, "html.parser")
    table = soup.findAll("table", {"class":"table-qsp-stats Mt(10px)"})[7]
    rows = table.findAll("tr")
    
    useful_string = filepath + ticker[i] + " characteristics.csv"
    with open(useful_string, "wt+", newline="", encoding = 'utf8') as f:
        writer = csv.writer(f)
        for row in rows:
            csv_row = []
            for cell in row.findAll(["td", "th"]):
                csv_row.append(cell.get_text())
            writer.writerow(csv_row)


# In[ ]:


"""high_corr = []
low_corr = []
for j in range(len(ticker)):
    df = pd.read_csv(filepath + ticker[j] + " characteristics.csv", names = ['Statistics', 'Values'])
    values = df['Values']
    if float(values.iloc[0]) > 1:
        high_corr.append(ticker[j])
    else:
        low_corr.append(ticker[j])
print (low_corr, high_corr)"""


# In[ ]:


def file_downloader(ticker):
    browser = splinter.browser.ChromeWebDriver()

    url = "https://finance.yahoo.com/quote/" + ticker + "/history?p=" + ticker
    browser.visit(url)
    button = browser.find_by_css('input[data-test="date-picker-full-range"]')
    button.click()
    time.sleep(1)
    date_fill = browser.find_by_css('input[name="startDate"]').fill(start_date)
    time.sleep(1)    
    button2 = browser.find_by_css('button[class=" Bgc($c-fuji-blue-1-b) Bdrs(3px) Px(20px) Miw(100px) Whs(nw) Fz(s) Fw(500) C(white) Bgc($actionBlueHover):h Bd(0) D(ib) Cur(p) Td(n)  Py(9px) Miw(80px)! Fl(start)"]').click()
    time.sleep(1)
    button3 = browser.find_by_css('button[class=" Bgc($c-fuji-blue-1-b) Bdrs(3px) Px(20px) Miw(100px) Whs(nw) Fz(s) Fw(500) C(white) Bgc($actionBlueHover):h Bd(0) D(ib) Cur(p) Td(n)  Py(9px) Fl(end)"]').click()
    time.sleep(1)
    button4 = browser.find_by_css('a[download="' + ticker + '.csv"]').click()
    time.sleep(1.5)
    browser.quit()
    return None


# In[ ]:


for i in range(len(ticker)):
    file_downloader(ticker[i])
    
stock_correl = correl(ticker)

def my_LSTM(ticker):
#read the file
    df = pd.read_csv(filepath + ticker + '.csv')
    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.index = df['Date']
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'], label='Close Price history')
    
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    dataset = new_data.values

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if np.isnan(dataset[i,j]) == True:
                dataset[i,j] = 0

    train_factor = 0.8
    valid_factor = 0.2
    train_length = int(train_factor*len(new_data))
    valid_length = int(valid_factor*len(new_data))
    step_size = int(days)

    close_final = np.array([])
    for i in range(step_size):
        train = dataset[:train_length+i,:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(train)

        x_train, y_train = [], []
        for m in range(step_size,len(train)):
            x_train.append(scaled_data[m-step_size:m,0])
            y_train.append(scaled_data[m,0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        model = Sequential()
        model.add(LSTM(units=200, input_shape = (x_train.shape[1],1)))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        model.fit(x_train, y_train, epochs = 1, batch_size=1, verbose=2)

        valid = dataset[-step_size:, :].copy()
        inputs = scaler.transform(valid)

        X_test = inputs.copy()
        X_test = np.array(X_test)
        X_test = X_test.reshape(1,*X_test.shape)
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)
        print (closing_price)

        close_final = np.append(close_final, closing_price)
        dataset = np.append(dataset, closing_price)
        dataset = dataset.reshape(len(dataset), 1)

        print ('Iteration ' + str(i) + ' complete')

    close_final = pd.Series(close_final)
    close_final.index = list(range(len(train),len(train)+step_size))
    close_final.to_csv(filepath + ticker + ' predictions.csv')

    new_train = new_data[:train_length+(2*step_size)].values
    plt.figure(figsize=(16,8))
    plt.plot(new_train)
    plt.plot(close_final)
    
    return True


# In[ ]:


for l in range(len(ticker)):
    my_LSTM(ticker[l])
    
"""def correl_for_pred(mylist):
    if len(mylist) == 1:
        print ('Need 2 stocks to correlate')
        return None
    correl_list = []
    for i in range(len(mylist)):
        if i == len(mylist) - 1:
            break
        j = i + 1
        flag = True
        while flag:
            df1 = pd.read_csv(filepath + mylist[i] + " predictions.csv", index_col = 0, names = ['Day', 'Predicted Close'])
            df2 = pd.read_csv(filepath + mylist[j] + " predictions.csv", index_col = 0, names = ['Day', 'Predicted Close'])
            list1 = df1['Predicted Close']
            list2 = df2['Predicted Close']

            numerator = 0
            d1 = 0
            d2 = 0
            list1_mean = np.mean(list1)
            list2_mean = np.mean(list2)
            for k in range(len(list1)):
                numerator += (list1[k] - list1_mean)*(list2[k] - list2_mean)
                d1 += (list1[k] - list1_mean)**2
                d2 += (list2[k] - list2_mean)**2
            stock_correl = numerator/np.sqrt(d1*d2)
            print (mylist[i] + ' predictions and ' + mylist[j] + ' predictions are correlated by ' + str(stock_correl))
            j += 1
            if j == len(mylist):
                flag = False
            correl_list.append(stock_correl)
    return correl_list

stock_pred_list = ticker
correl_for_pred(stock_pred_list)"""

for index in ticker:
    os.remove('C:\\Users\\Shaurya\\Downloads\\' + index + '.csv')