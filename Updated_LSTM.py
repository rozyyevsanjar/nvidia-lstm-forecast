#!/usr/bin/env python
# coding: utf-8

# import statements
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import feedparser
import plotly.graph_objects as go


# In[440]:


# Config
days_to_fetch = 30
max_articles_per_day = 3
all_articles = []
analyzer = SentimentIntensityAnalyzer()

for i in range(days_to_fetch):
    day = datetime.today() - timedelta(days=i)
    date_str = day.strftime('%Y-%m-%d')

    # Google News RSS URL (NVIDIA news from i days ago)
    url = f"https://news.google.com/rss/search?q=nvidia+when:{i+1}d&hl=en&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    entries = feed.entries[:max_articles_per_day]

    for entry in entries:
        title = entry.get('title', '')
        desc = entry.get('summary', '')
        combined = f"{title}. {desc}"

        sentiment = analyzer.polarity_scores(combined)['compound']
        all_articles.append((date_str, title, desc, sentiment))

# Create DataFrame
news_df = pd.DataFrame(all_articles, columns=['Date', 'Headline', 'Description', 'Sentiment'])
news_df['Date'] = pd.to_datetime(news_df['Date'])

# Daily average sentiment
daily_sentiment = news_df.groupby('Date')['Sentiment'].mean().to_frame()

# ✅ Print result
print("✅ Google News sentiment collected for", len(daily_sentiment), "days")
print(daily_sentiment.head())


# In[441]:

def news_sentiment(daily_sentiment):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_sentiment.index, daily_sentiment['Sentiment'], marker='o', linestyle='-', color='white')
    ax.set_title("📊 NVIDIA News Sentiment Over Time (Title + Description)", color='white', fontsize='xx-large')
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Sentiment Score")
    ax.axhline(0, color='blue', linestyle='--', linewidth=1)
    ax.grid(True)
    fig.tight_layout()
    return fig


# In[442]:


print(news_df.sort_values('Date').groupby('Date').head(3)[['Date', 'Headline', 'Sentiment']])


# In[443]:


# Step 1: Download NVIDIA stock data
end_date = datetime.today()
start_date = end_date - timedelta(days=3*365)  # 3 years of data
nvidia_data = yf.download('NVDA', start=start_date, end=end_date)

# Check and flatten MultiIndex
# Flatten MultiIndex columns from yfinance
if isinstance(nvidia_data.columns, pd.MultiIndex):
    nvidia_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in nvidia_data.columns]


# In[444]:


nvidia_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']


# In[445]:


nvidia_data.describe()


# In[446]:


nvidia_data.tail()


# In[447]:


nvidia_data.info()


# In[448]:

def stock_price_plotly(nvidia_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nvidia_data.index,
        y=nvidia_data['Close'],
        mode='lines',
        name='NVIDIA Closing Price',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title="📈 NVIDIA Stock Price Over Time (Last 3 Years)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )
    return fig

# In[449]:


nvidia_data.index = pd.to_datetime(nvidia_data.index)
daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

# Make sure column name is correct
daily_sentiment.columns = ['Sentiment']

# Drop any existing 'Sentiment' column
if 'Sentiment' in nvidia_data.columns:
    nvidia_data = nvidia_data.drop(columns='Sentiment')

# Merge cleanly
nvidia_data = nvidia_data.merge(daily_sentiment, left_index=True, right_index=True, how='left')
nvidia_data['Sentiment'] = nvidia_data['Sentiment'].ffill()

# Verify
print(nvidia_data[['Close', 'Sentiment']].tail())


# In[450]:


nvidia_data['DayOfWeek'] = nvidia_data.index.dayofweek  # 0 = Monday
nvidia_data['Month'] = nvidia_data.index.month

feature_cols = ['Close', 'Volume', 'Sentiment', 'DayOfWeek', 'Month']
# In[451]:


# Step 1: Only fill missing sentiment (leave prices as-is)
nvidia_data['Sentiment'] = nvidia_data['Sentiment'].ffill().bfill()

# Step 2: Then extract features (drop rows only if any price data is missing — optional)
data = nvidia_data[feature_cols]

# Optional safety: drop rows where essential price columns are NaN
data = data.dropna(subset=['Close', 'Volume'])

# Baseline (no seasonality, no sentiment)
baseline_cols = ['Close', 'Volume']
data_baseline = nvidia_data[baseline_cols].dropna()

# Seasonality only
seasonality_cols = ['Close', 'Volume', 'DayOfWeek', 'Month']
data_seasonality = nvidia_data[seasonality_cols].dropna()

# Sentiment only
sentiment_cols = ['Close', 'Volume', 'Sentiment']
data_sentiment = nvidia_data[sentiment_cols].dropna()

features = ['Close', 'Volume']


# Build the dataset
data = nvidia_data[features].dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 60  # or 90 — depending on your LSTM setup

# Rebuild sequences
def create_sequences(data, sequence_length=90):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

x_all, y_all = create_sequences(scaled_data, sequence_length)


# In[452]:


nvidia_data.head()


# In[453]:


print("x_all shape:", x_all.shape)
print("y_all shape:", y_all.shape)


# In[454]:


print(data.head())
print(data.shape)


# === Sequence builder ===
def create_sequences(data, sequence_length=60):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # 'Close' is assumed to be the first column
    return np.array(x), np.array(y)

# === LSTM Runner Function with Early Stopping ===
def run_lstm_with_features(nvidia_data, selected_features, sequence_length=60):
    # Step 1: Drop NaNs based on selected features
    data = nvidia_data[selected_features].dropna()

    # Step 2: Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Step 3: Create sequences
    x_all, y_all = create_sequences(scaled_data, sequence_length)

    # Step 4: Train/test split (80/20)
    split = int(0.8 * len(x_all))
    x_train, x_test = x_all[:split], x_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    # Step 5: Define Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',    # Watch validation loss
        patience=5,            # Stop if no improvement for 5 epochs
        restore_best_weights=True  # Go back to best-performing weights
    )

    # Step 6: Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Step 7: Predict
    predicted = model.predict(x_test)

    # Step 8: Inverse scale predictions
    predicted_prices = scaler.inverse_transform(
        np.hstack((predicted, np.zeros((predicted.shape[0], scaled_data.shape[1] - 1))))
    )[:, 0]

    actual_prices = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))))
    )[:, 0]

    # Step 9: Metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    r2 = r2_score(actual_prices, predicted_prices)

    return predicted_prices, actual_prices, mae, rmse, r2


# In[466]:


# ✅ Get the full list of original dates from your preprocessed `data`
original_dates = data.index

# ✅ Get the test dates (align with y_test after sequence + split)
test_dates = original_dates[sequence_length + split:]

# ✅ Make sure they match
print("Test dates shape:", test_dates.shape)
print("Actual prices shape:", actual_prices.shape)

# Assumes `nvidia_data` and `run_lstm_with_features()` are already available

# === Define all feature sets ===
baseline_features = ['Close', 'Volume']
seasonality_features = ['Close', 'Volume', 'DayOfWeek', 'Month']
sentiment_features = ['Close', 'Volume', 'Sentiment']
full_features = ['Close', 'Volume', 'Sentiment', 'DayOfWeek', 'Month']

# === Generate predictions for each model ===
pred_baseline, actual_baseline, mae_b, rmse_b, r2_b = run_lstm_with_features(nvidia_data, baseline_features)
pred_seasonal, actual_seasonal, mae_s, rmse_s, r2_s = run_lstm_with_features(nvidia_data, seasonality_features)
pred_sentiment, actual_sentiment, mae_sent, rmse_sent, r2_sent = run_lstm_with_features(nvidia_data, sentiment_features)
pred_full, actual_full, mae_f, rmse_f, r2_f = run_lstm_with_features(nvidia_data, full_features)

# === Function to select predictions by model name ===
def get_model_predictions(model_type):
    if model_type == "Baseline":
        return pred_baseline, actual_baseline, mae_b, rmse_b, r2_b
    elif model_type == "Seasonality Only":
        return pred_seasonal, actual_seasonal, mae_s, rmse_s, r2_s
    elif model_type == "Sentiment Only":
        return pred_sentiment, actual_sentiment, mae_sent, rmse_sent, r2_sent
    elif model_type == "Full (Sentiment + Seasonality)":
        return pred_full, actual_full, mae_f, rmse_f, r2_f
    else:
        raise ValueError("Invalid model type provided.")

# In[468]:

def plot_forecast_comparison(test_dates, actual_prices,
                             pred_baseline, pred_seasonality,
                             pred_sentiment, pred_full):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test_dates, y=actual_prices,
        mode='lines', name='Actual',
        line=dict(width=2, color='black')
    ))

    fig.add_trace(go.Scatter(
        x=test_dates, y=pred_baseline,
        mode='lines', name='LSTM (No Seasonality/Sentiment)',
        line=dict(dash='dash', color='gray')
    ))

    fig.add_trace(go.Scatter(
        x=test_dates, y=pred_seasonality,
        mode='lines', name='LSTM + Seasonality',
        line=dict(dash='dot', color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=test_dates, y=pred_sentiment,
        mode='lines', name='LSTM + Sentiment',
        line=dict(dash='dash', color='green')
    ))

    fig.add_trace(go.Scatter(
        x=test_dates, y=pred_full,
        mode='lines', name='LSTM + Sentiment + Seasonality',
        line=dict(dash='solid', color='blue')
    ))

    fig.update_layout(
        title="📊 LSTM Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99)
    )

    return fig

# In[476]:


residuals = actual_prices - predicted_prices


# In[478]:

def residual_plot(residuals):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_dates, residuals, marker='o', linestyle='-', color='crimson')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("🔍 Residuals (Actual - Predicted) for Test Set")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual ($)")
    ax.grid(True)
    fig.tight_layout()
    return fig


# In[480]:


plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
plt.title("📊 Residual Distribution")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




