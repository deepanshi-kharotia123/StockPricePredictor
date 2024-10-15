import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model 
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")
stock=st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end= datetime.now()
start=datetime(end.year-20, end.month, end.day)
google_data=yf.download(stock, start, end)


model= load_model("Latest_stock_price_model.keras") 
st.subheader("Stock Data (yfinance)")
st.write(google_data)

splitting_len= int(len(google_data)*0.7)
x_test= pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset = None):
    fig= plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close,'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days']=google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days']=google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days']=google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])
x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)



st.subheader("Future Stock Price Predictions")

num_future_days = 30

last_100_days = scaled_data[-100:]
future_predictions = []

for i in range(num_future_days):
    X_future = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))
    future_price = model.predict(X_future)
    future_predictions.append(future_price[0, 0])

    last_100_days = np.append(last_100_days[1:], future_price, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

future_dates = pd.date_range(google_data.index[-1], periods=num_future_days)

future_data = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])

fig1 = plt.figure(figsize=(15,6))
plt.plot(future_data.index, future_data['Predicted Close'], color='green', linestyle='-', marker='o')  
plt.xlabel('Date')
plt.ylabel('Predicted Close Price')
plt.title('Predicted Future Stock Prices')
st.pyplot(fig1)

st.write(future_data)