#!/usr/bin/env python
# coding: utf-8

# === Imports ===
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

# === Step 1: Collect Sentiment Data ===
days_to_fetch = 30
max_articles_per_day = 3
all_articles = []
analyzer = SentimentIntensityAnalyzer()

for i in range(days_to_fetch):
    day = datetime.today() - timedelta(days=i)
    date_str = day.strftime('%Y-%m-%d')
    url = f"https://news.google.com/rss/search?q=nvidia+when:{i+1}d&hl=en&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    entries = feed.entries[:max_articles_per_day]
    for entry in entries:
        title = entry.get('title', '')
        desc = entry.get('summary', '')
        combined = f"{title}. {desc}"
        sentiment = analyzer.polarity_scores(combined)['compound']
        all_articles.append((date_str, title, desc, sentiment))

news_df = pd.DataFrame(all_articles, columns=['Date', 'Headline', 'Description', 'Sentiment'])
news_df['Date'] = pd.to_datetime(news_df['Date'])
daily_sentiment = news_df.groupby('Date')['Sentiment'].mean().to_frame()


def news_sentiment(daily_sentiment):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sentiment.index,
        y=daily_sentiment['Sentiment'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='steelblue'),
        marker=dict(size=6)
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.update_layout(
        title="📊 NVIDIA News Sentiment Over Time (Title + Description)",
        xaxis_title="Date",
        yaxis_title="Average Sentiment Score",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# === Step 2: Get Stock Data ===
end_date = datetime.today()
start_date = end_date - timedelta(days=3*365)
nvidia_data = yf.download('NVDA', start=start_date, end=end_date)
if isinstance(nvidia_data.columns, pd.MultiIndex):
    nvidia_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in nvidia_data.columns]
nvidia_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# === Step 3: Merge Sentiment and Add Features ===
daily_sentiment.columns = ['Sentiment']
nvidia_data.index = pd.to_datetime(nvidia_data.index)
daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
nvidia_data = nvidia_data.drop(columns='Sentiment', errors='ignore')
nvidia_data = nvidia_data.merge(daily_sentiment, left_index=True, right_index=True, how='left')
nvidia_data['Sentiment'] = nvidia_data['Sentiment'].ffill().bfill()
nvidia_data['DayOfWeek'] = nvidia_data.index.dayofweek
nvidia_data['Month'] = nvidia_data.index.month

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
# === Step 4: Model Utilities ===
def create_sequences(data, sequence_length=60):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def run_lstm_with_features(nvidia_data, selected_features, sequence_length=60):
    data = nvidia_data[selected_features].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    x_all, y_all = create_sequences(scaled_data, sequence_length)
    split = int(0.8 * len(x_all))
    x_train, x_test = x_all[:split], x_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    predicted = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(np.hstack((predicted, np.zeros((predicted.shape[0], scaled_data.shape[1] - 1)))))[:, 0]
    actual_prices = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1)))))[:, 0]

    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    r2 = r2_score(actual_prices, predicted_prices)

    return predicted_prices, actual_prices, mae, rmse, r2, scaled_data, y_all, split

def predict_tomorrow(model, scaler, scaled_data, rmse, sequence_length=60):
    """
    Predict the next day's price using the trained model and most recent data.

    Parameters:
        model (keras.Model): Trained LSTM model.
        scaler (MinMaxScaler): Fitted scaler used during training.
        scaled_data (ndarray): The full scaled dataset used to train the model.
        rmse (float): RMSE from test set to estimate the prediction range.
        sequence_length (int): Number of time steps the model expects.

    Returns:
        predicted_price (float): The predicted price for tomorrow.
        lower_bound (float): Lower bound using RMSE.
        upper_bound (float): Upper bound using RMSE.
    """
    # 1. Get the most recent sequence
    last_sequence = scaled_data[-sequence_length:]
    input_data = np.expand_dims(last_sequence, axis=0)

    # 2. Predict the next step
    predicted_scaled = model.predict(input_data)

    # 3. Inverse scale to original price
    predicted_price = scaler.inverse_transform(
        np.hstack((predicted_scaled, np.zeros((1, scaled_data.shape[1] - 1))))
    )[0, 0]

    # 4. Create prediction range using RMSE
    lower_bound = predicted_price - rmse
    upper_bound = predicted_price + rmse

    return predicted_price, lower_bound, upper_bound





# === Step 5: Define Feature Sets and Run Models ===
baseline_features = ['Close', 'Volume']
seasonality_features = ['Close', 'Volume', 'DayOfWeek', 'Month']
sentiment_features = ['Close', 'Volume', 'Sentiment']
full_features = ['Close', 'Volume', 'Sentiment', 'DayOfWeek', 'Month']

pred_baseline, actual_baseline, mae_b, rmse_b, r2_b, scaled_baseline, y_all_baseline, split = run_lstm_with_features(nvidia_data, baseline_features)
pred_seasonal, actual_seasonal, mae_s, rmse_s, r2_s, _, _, _ = run_lstm_with_features(nvidia_data, seasonality_features)
pred_sentiment, actual_sentiment, mae_sent, rmse_sent, r2_sent, _, _, _ = run_lstm_with_features(nvidia_data, sentiment_features)
pred_full, actual_full, mae_f, rmse_f, r2_f, _, y_all_full, _ = run_lstm_with_features(nvidia_data, full_features)

# === Prepare Dates ===
sequence_length = 60
data = nvidia_data[baseline_features].dropna()
original_dates = data.index
all_dates = original_dates[sequence_length:]
test_dates = all_dates[split:]

# === Forecast Comparison Plot ===
def plot_forecast_comparison(test_dates, actual_prices,
                             pred_baseline, pred_seasonality,
                             pred_sentiment, pred_full,
                             show_baseline=True, show_seasonal=True,
                             show_sentiment=True, show_full=True):
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_dates, y=actual_prices,
        mode='lines', name='Actual',
        line=dict(width=2, color='purple')
    ))

    if show_baseline:
        fig.add_trace(go.Scatter(
            x=test_dates, y=pred_baseline,
            mode='lines', name='LSTM (No Seasonality/Sentiment)',
            line=dict(dash='dash', color='gray')
        ))

    if show_seasonal:
        fig.add_trace(go.Scatter(
            x=test_dates, y=pred_seasonality,
            mode='lines', name='LSTM + Seasonality',
            line=dict(dash='dot', color='orange')
        ))

    if show_sentiment:
        fig.add_trace(go.Scatter(
            x=test_dates, y=pred_sentiment,
            mode='lines', name='LSTM + Sentiment',
            line=dict(dash='dash', color='green')
        ))

    if show_full:
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
        hovermode="x unified"
    )
    return fig

def plot_weekly_seasonality(nvidia_data):
    weekly_avg = nvidia_data.groupby('DayOfWeek')['Close'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[days[d] for d in weekly_avg.index],
        y=weekly_avg.values,
        marker_color='mediumseagreen'
    ))

    fig.update_layout(
        title="📅 Average Closing Price by Day of the Week",
        xaxis_title="Day of Week",
        yaxis_title="Average Close Price (USD)",
        template="plotly_white"
    )
    return fig

def plot_monthly_seasonality(nvidia_data):
    monthly_avg = nvidia_data.groupby('Month')['Close'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[months[m-1] for m in monthly_avg.index],
        y=monthly_avg.values,
        marker_color='cornflowerblue'
    ))

    fig.update_layout(
        title="📆 Average Closing Price by Month",
        xaxis_title="Month",
        yaxis_title="Average Close Price (USD)",
        template="plotly_white"
    )
    return fig

def residual_plot(test_dates, residuals):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=residuals,
        mode='lines+markers',
        name='Residuals',
        line=dict(color='crimson')
    ))
    fig.add_hline(y=0, line_dash="dash", line_color='gray')

    fig.update_layout(
        title="🔍 Residuals (Actual - Predicted)",
        xaxis_title="Date",
        yaxis_title="Residual ($)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def residual_distribution(residuals):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color='skyblue',
        opacity=0.85,
        name='Residuals'
    ))

    fig.update_layout(
        title="📊 Residual Distribution (Plotly)",
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Frequency",
        template="plotly_white",
        bargap=0.05
    )

    return fig

def get_today_headlines():
    today_str = datetime.today().strftime('%Y-%m-%d')
    return news_df[news_df['Date'] == today_str].head(3)

def get_model_and_scaler(nvidia_data, selected_features, sequence_length=60):
    """
    Trains an LSTM model on full data with selected features.
    Returns trained model, fitted scaler, and scaled dataset.
    """
    data = nvidia_data[selected_features].dropna()

    # Scale input features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    x_all, y_all = create_sequences(scaled_data, sequence_length)

    # Define model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_all.shape[1], x_all.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train on entire dataset
    model.fit(x_all, y_all, epochs=100, batch_size=32, verbose=0)

    return model, scaler, scaled_data

def predict_future_lstm(model, scaler, scaled_data, forecast_days=1, sequence_length=60):
    predictions = []

    current_sequence = scaled_data[-sequence_length:]

    for _ in range(forecast_days):
        input_data = np.expand_dims(current_sequence, axis=0)
        predicted_scaled = model.predict(input_data)
        
        # Append prediction
        predictions.append(predicted_scaled[0, 0])

        # Update sequence by appending new prediction
        next_step = np.hstack((predicted_scaled, np.zeros((1, scaled_data.shape[1] - 1))))
        next_scaled = scaler.inverse_transform(next_step)[0]
        next_scaled = scaler.transform([next_scaled])  # Rescale
        current_sequence = np.vstack((current_sequence[1:], next_scaled))

    # Inverse scale predictions
    inv_predictions = scaler.inverse_transform(
        np.hstack((np.array(predictions).reshape(-1, 1), np.zeros((forecast_days, scaled_data.shape[1] - 1))))
    )[:, 0]

    return inv_predictions
