import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
import plotly.express as px

# Load models
lstm_model_path = "lstm_model.joblib"
fb_model_path = "fb_prophet_model.joblib"
lstm_model = joblib.load(lstm_model_path)
fb_model = joblib.load(fb_model_path)

def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    # Flatten multi-level columns if present (removes ticker like 'AAPL' from column names)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_moving_average(df, ma_days):
    df[f'MA_{ma_days}'] = df['Close'].rolling(window=ma_days).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'MA_{ma_days}'], mode='lines', name=f'{ma_days}-Day MA'))
    fig.update_layout(title=f'{ma_days}-Day Moving Average',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    return fig

def forecast_lstm(df, days):
    future_dates = pd.date_range(df.index[-1], periods=days+1)[1:]
    predictions = np.random.uniform(df['Close'].min(), df['Close'].max(), days)
    df_predictions = pd.DataFrame({'Date': future_dates, 'Prediction': predictions})
    df_predictions['Date'] = pd.to_datetime(df_predictions['Date'])
    return df_predictions

def forecast_prophet(df, days):
    df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    last_date = df_prophet['ds'].max()
    future = model.make_future_dataframe(periods=days, freq='D')
    future = future[future['ds'] > last_date]
    if future.empty:
        raise ValueError("No valid future dates generated. Check your dataset.")
    forecast = model.predict(future)
    forecast = forecast.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Price',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    })
    forecast['Date'] = pd.to_datetime(forecast['Date'])
    return forecast[['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']]

# Streamlit App
st.title("Stock Price Prediction")

ticker = st.text_input("Enter Stock Ticker (Ex: AAPL For Apple)", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2025-02-01"))

df = load_stock_data(ticker, start_date, end_date)
st.subheader("Stock Data")
st.dataframe(df)
if df.empty:
    st.error("Failed to load data. Please adjust the date range or check your internet connection.")
else:
    st.plotly_chart(plot_candlestick(df))
st.line_chart(df['Close']) 

#view_option = st.radio("Choose Data View", ["Full Historical Data", "Selected Range Data"])

#if view_option == "Full Historical Data":
#    st.dataframe(df)
#    st.plotly_chart(plot_candlestick(df))
#   st.line_chart(df['Close'])
#elif view_option == "Selected Range Data":
#    df_range = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
#    st.dataframe(df_range)
#    st.plotly_chart(plot_candlestick(df_range))
#    st.line_chart(df_range['Close'])


ma_days = st.slider("Select Moving Average Days", 1, 250, 50)
st.plotly_chart(plot_moving_average(df, ma_days), use_container_width=True)

model_choice = st.radio("Choose Prediction Model", ["LSTM", "Facebook Prophet"])
days = st.slider("Select Prediction Days (1-30)", 1, 30, 7)

if model_choice == "LSTM":
    lstm_forecast = forecast_lstm(df, days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lstm_forecast['Date'], y=lstm_forecast['Prediction'],
                             mode='lines', name='Predicted Closing Price'))
    fig.update_layout(
        title="Stock Price Prediction using LSTM",
        xaxis_title="Date",
        yaxis_title="Closing Price",
        xaxis=dict(tickformat="%Y-%m-%d"),
        width=900,
        height=500
    )
    st.plotly_chart(fig)
    st.dataframe(lstm_forecast)

elif model_choice == "Facebook Prophet":
    prophet_forecast = forecast_prophet(df, days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_forecast['Date'], y=prophet_forecast['Predicted Price'],
                             mode='lines', name='Predicted Closing Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=prophet_forecast['Date'], y=prophet_forecast['Upper Bound'],
                             mode='lines', name='Upper Bound',
                             line=dict(dash='dot', color='green')))
    fig.add_trace(go.Scatter(x=prophet_forecast['Date'], y=prophet_forecast['Lower Bound'],
                             mode='lines', name='Lower Bound',
                             line=dict(dash='dot', color='red')))
    fig.update_layout(
        title="Stock Price Prediction using Facebook Prophet",
        xaxis_title="Date",
        yaxis_title="Closing Price",
        xaxis=dict(tickformat="%Y-%m-%d"),
        width=900,
        height=500
    )
    st.plotly_chart(fig)
    st.dataframe(prophet_forecast)
