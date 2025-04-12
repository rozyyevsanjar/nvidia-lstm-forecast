# 🧠 NVIDIA Stock Forecasting Dashboard

A real-time dashboard that forecasts NVIDIA stock prices using an LSTM model powered by historical stock data and live sentiment analysis from Google News. Built with 💻 Streamlit + 📈 Plotly.

---

## 🚀 Features

- 📊 LSTM-based forecasting model
- 📰 Real-time sentiment analysis from Google News
- 📈 Seasonality features and analysis
- 📅 Forecast horizon selector (1–5 days)
- 📉 Residual & error analysis
- 📱 Fully responsive
- 📎 Download charts + data

---

## 🔍 Tech Stack

| Tool           | Purpose                                |
|----------------|----------------------------------------|
| `Python`       | Core logic + ML                        |
| `Streamlit`    | Interactive frontend UI                |
| `Plotly`       | Charts + visualizations                |
| `yfinance`     | Stock data API                         |
| `VADER`        | Sentiment analysis (Google News)       |
| `LSTM (Keras)` | Sequential forecasting model           |
| `scikit-learn` | Preprocessing & evaluation             |
| `Kaleido`      | Exporting charts as PNG                |

---

## 📦 Installation (Local)

Clone the repo and run it locally:

```bash
git clone https://github.com/yourusername/nvidia-lstm-forecast.git
cd nvidia-lstm-forecast
pip install -r requirements.txt
streamlit run app.py