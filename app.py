import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load the pre-trained model
model = load_model(r'C:\Users\Admin\Desktop\Stock Price\Stock Predictions Model.keras')

# Streamlit header
st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Date input for date range selection
start_date = st.date_input('Select start date', datetime(2012, 1, 1))
end_date = st.date_input('Select end date', datetime(2024, 10, 19))

# Ensure the selected end date is not earlier than the start date
if start_date > end_date:
    st.error('Error: End date must fall after start date.')

# Download stock data
data = yf.download(stock, start=start_date, end=end_date)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Split data into training and testing sets
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Min-Max Scaling
scaler = MinMaxScaler(feature_range=(0,1))

# Concatenate last 100 days of training data to test set for scaling
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig2)

# Plot Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Make predictions
predict = model.predict(x)

# Reverse scaling of predicted and original prices using inverse_transform
predict = scaler.inverse_transform(predict)
y = scaler.inverse_transform(y.reshape(-1, 1))

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y, 'g', label='Original Price')  # Ensure this is plotted first (original values)
plt.plot(predict, 'r', label='Predicted Price')  # Plot predicted prices after reversing scaling
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Button for Next Day Prediction
if st.button('Predict Next Day Price'):
    # Prepare the last 100 days of data for the next day prediction
    last_100_days = data_test_scale[-100:]
    last_100_days = np.reshape(last_100_days, (1, last_100_days.shape[0], last_100_days.shape[1]))

    # Predict the next day's price
    next_day_pred = model.predict(last_100_days)

    # Reverse scaling
    next_day_pred = scaler.inverse_transform(next_day_pred)

    # Display the predicted next day price
    st.subheader('Next Day Predicted Price')
    st.text(f'The predicted price for the next day is: {next_day_pred[0][0]:.2f}')