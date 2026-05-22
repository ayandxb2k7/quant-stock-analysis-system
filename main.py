import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fetch stock data
stock = "AAPL"

data = yf.download(stock, start="2022-01-01", end="2025-01-01")

# Moving averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Prepare data for prediction
data = data.dropna()

X = np.array(range(len(data))).reshape(-1, 1)
y = data['Close'].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future prices
future_days = 30
future_X = np.array(range(len(data) + future_days)).reshape(-1, 1)
predictions = model.predict(future_X)

# Plot
plt.figure(figsize=(12,6))

plt.plot(data.index, data['Close'], label='Actual Price')
plt.plot(data.index, data['MA50'], label='50 Day MA')
plt.plot(data.index, data['MA200'], label='200 Day MA')

future_dates = pd.date_range(start=data.index[-1], periods=future_days+1)

plt.plot(
    list(data.index) + list(future_dates[1:]),
    predictions,
    label='Predicted Trend'
)

plt.title(f"{stock} Stock Analysis")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

plt.show()

print("\nStock Analysis Completed Successfully!")