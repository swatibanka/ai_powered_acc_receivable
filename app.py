import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import datetime as dt
from datetime import date, timedelta

df = pd.read_csv('Accounts-Receivable.csv')
df1 = pd.read_csv('Accounts-Receivable.csv')

#print(df)

# chnage the type of column
df['countryCode'] = df['countryCode'].astype(object)
df['invoiceNumber'] = df['invoiceNumber'].astype(object)

# We will make a new column that replaces the country code with the name of the country that may be useful in our analysis
df['countryName'] = df['countryCode'].replace({ 391:'Germany',406:'Australia',818:'California, US',897:'Kansas, US',770:'Georgia, US' })
df.drop('countryCode',axis = 1,inplace = True)

# Changing the type of columns to use pandas time properties
df['PaperlessDate'] = pd.to_datetime(df['PaperlessDate'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['DueDate'] = pd.to_datetime(df['DueDate'])
df['SettledDate'] = pd.to_datetime(df['SettledDate'])

date_columns = ['PaperlessDate','InvoiceDate','DueDate','SettledDate']

df['PaperlessDate_year'] = df['PaperlessDate'].dt.year
df['PaperlessDate_day'] = df['PaperlessDate'].dt.day
df['PaperlessDate_month'] = df['PaperlessDate'].dt.month

df['InvoiceDate_year'] = df['InvoiceDate'].dt.year
df['InvoiceDate_month'] = df['InvoiceDate'].dt.month
df['InvoiceDate_day'] = df['InvoiceDate'].dt.day

df['DueDate_year'] = df['DueDate'].dt.year
df['DueDate_month'] = df['DueDate'].dt.month
df['DueDate_day'] = df['DueDate'].dt.day

df['SettledDate_year'] = df['SettledDate'].dt.year
df['SettledDate_month'] = df['SettledDate'].dt.month
df['SettledDate_day'] = df['SettledDate'].dt.day

# Remove an unimportant column
df.drop(['customerID','invoiceNumber','SettledDate','InvoiceDate','PaperlessDate','DueDate'], axis = 1, inplace = True) 

#Encoding

category = ['countryName'] 
encoded_categ = pd.get_dummies(df[category] ,drop_first=True)

df['Disputed'] = df['Disputed'].replace({'Yes':1, 'No':0})
df['PaperlessBill'] = df['PaperlessBill'].replace({'Electronic':1, 'Paper':0})

#Likning the encoed_cateh with the df
df = pd.concat([df, encoded_categ], axis = 1)
#print(df)


# Dropping the categorical features
df = df.drop(columns = category, axis = 1)

#Now let's modify the type of columns
columns = list(df.select_dtypes(['object','uint8']).columns)

for i in columns:
    df[i] = df[i].astype('int64')
df.dtypes.to_frame(name='Type')

df.drop('InvoiceDate_day', axis =1, inplace = True)

#print(df)

# Scale test data
#scaler = StandardScaler()
#X_test = scaler.fit_transform(df)

#model pickle
model_new = pickle.load(open("model.pkl", "rb"))

pred_dayslate = model_new.predict(df)

#print(pred_dayslate)
print(type(pred_dayslate))
my_array = np.array(pred_dayslate)
df2 = pd.DataFrame(my_array,columns=['preddays'])
#print(type(df2))
#print(type(df1))


result = pd.concat([df1,df2] ,axis=1, join='inner')
#print(result)

# Create flask app
app = Flask(__name__)


@app.route("/", methods = ["POST","GET"])
def Home():
    return jsonify("Hello World")

@app.route("/predict_dayslate", methods = ["GET","POST"])
def predict_dayslate():
    json_data = result.to_json()
    return (json_data)

if __name__ == "__main__":
        app.run()

