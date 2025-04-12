import streamlit as st
import pandas as pd
from Updated_LSTM1 import (
    run_lstm_with_features, plot_forecast_comparison, 
    baseline_features, seasonality_features, sentiment_features, full_features,
    pred_baseline, actual_baseline, mae_b, rmse_b, r2_b,
    pred_seasonal, actual_seasonal, mae_s, rmse_s, r2_s,
    pred_sentiment, actual_sentiment, mae_sent, rmse_sent, r2_sent,
    pred_full, actual_full, mae_f, rmse_f, r2_f,
    test_dates, news_sentiment, stock_price_plotly, nvidia_data, daily_sentiment, 
    plot_monthly_seasonality, plot_weekly_seasonality, residual_plot, residual_distribution,
    get_today_headlines, predict_future_lstm, get_model_and_scaler
)

st.set_page_config(page_title="NVIDIA LSTM Forecast Dashboard", layout="wide")
st.title("📊 NVIDIA Stock Forecasting with LSTM")

# === Section: Today's Top Headlines ===
st.subheader("🗞️ Top 3 Headlines for Today")
today_headlines = get_today_headlines()
if today_headlines.empty:
    st.info("No headlines available yet for today.")
else:
    for _, row in today_headlines.iterrows():
        st.markdown(f"- [{row['Headline']}]({'https://news.google.com/search?q=' + row['Headline'].replace(' ', '+')})")

# === Section: Sentiment Chart ===
st.subheader("📰 News Sentiment Over Time")
sentiment_chart = news_sentiment(daily_sentiment)
st.plotly_chart(sentiment_chart)

# === Section: Seasonality Charts
st.subheader("🧭 Seasonality Insights")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(plot_weekly_seasonality(nvidia_data), use_container_width=True)

with col2:
    st.plotly_chart(plot_monthly_seasonality(nvidia_data), use_container_width=True)
# === Section: Historical Stock Price ===
st.subheader("📈 Historical NVIDIA Stock Price")
price_chart = stock_price_plotly(nvidia_data)
st.plotly_chart(price_chart, use_container_width=True)

st.subheader("🧠 Choose Models to Display")

show_baseline = st.checkbox("LSTM (No Seasonality/Sentiment)", value=True)
show_seasonal = st.checkbox("LSTM + Seasonality", value=True)
show_sentiment = st.checkbox("LSTM + Sentiment", value=True)
show_full = st.checkbox("LSTM + Sentiment + Seasonality", value=True)


# === Section: LSTM Model Comparison ===
st.subheader("🧠 Forecast Comparison")
comparison_fig = plot_forecast_comparison(
    test_dates, actual_full,
    pred_baseline, pred_seasonal,
    pred_sentiment, pred_full,
    show_baseline, show_seasonal, show_sentiment, show_full
)
st.plotly_chart(comparison_fig, use_container_width=True)


# === Section: Interactive LSTM Model Selector ===
model_choice = st.selectbox("🔍 Select LSTM Model Version", [
    "Baseline", 
    "Seasonality Only", 
    "Sentiment Only", 
    "Full (Sentiment + Seasonality)"
])

if model_choice == "Baseline":
    y_pred, y_true, mae, rmse, r2 = pred_baseline, actual_baseline, mae_b, rmse_b, r2_b
elif model_choice == "Seasonality Only":
    y_pred, y_true, mae, rmse, r2 = pred_seasonal, actual_seasonal, mae_s, rmse_s, r2_s
elif model_choice == "Sentiment Only":
    y_pred, y_true, mae, rmse, r2 = pred_sentiment, actual_sentiment, mae_sent, rmse_sent, r2_sent
else:
    y_pred, y_true, mae, rmse, r2 = pred_full, actual_full, mae_f, rmse_f, r2_f

# === Section: Metrics ===
st.subheader("📊 Model Evaluation Metrics")
st.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
st.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
st.metric("R² Score", f"{r2:.4f}")

# === Section: Residual Analysis ===
st.subheader("🧪 Residual Analysis")
residuals = y_true - y_pred
col1, col_space, col2 = st.columns([1, 0.1, 1])
with col1:
    st.plotly_chart(residual_plot(test_dates, residuals))
with col2:
    st.plotly_chart(residual_distribution(residuals))
# === Section: DataFrame for predictions ===
st.subheader("📋 Forecast Data Table")
df_result = pd.DataFrame({
    'Date': test_dates,
    'Actual': y_true,
    'Predicted': y_pred
})
df_result.set_index('Date', inplace=True)   
# === Section: DataFrame for predictions ===
show_df = st.checkbox("📋 Show Forecast Data", value=False)
if show_df:
    df_result = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_true,
        'Predicted': y_pred
    })
    st.dataframe(df_result.set_index('Date'))

    # === Download Buttons ===
# CSV export
csv = df_result.to_csv().encode('utf-8')
st.download_button(
    label="📥 Download Forecast Data (CSV)",
    data=csv,
    file_name='forecast_data.csv',
    mime='text/csv'
)

# PNG export of the comparison chart
img_bytes = comparison_fig.to_image(format="png")
st.download_button(
    label="📸 Download Forecast Chart (PNG)",
    data=img_bytes,
    file_name="forecast_chart.png",
    mime="image/png"
)
# Predicting future's forecast
st.subheader("📅 Forecast Window")
forecast_days = st.selectbox(
    "Select number of days to forecast:",
    options=[1, 3, 5],
    index=0
)
# Train your model (baseline + sentiment)
model, scaler, scaled_data = get_model_and_scaler(nvidia_data, ['Close', 'Volume', 'Sentiment'])

# Forecast
future_preds = predict_future_lstm(model, scaler, scaled_data, forecast_days)

# Display
st.subheader(f"🔮 Forecast for Next {forecast_days} Day(s) (BETA)")
for i, price in enumerate(future_preds, 1):
    st.metric(f"Day {i}", f"${price:.2f}")