import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# --- Set up the Streamlit App and Sidebar ---
st.set_page_config(layout="wide")
st.title("LSTM Model for Stock Price Prediction 📈")
st.markdown("This application predicts stock prices using a standalone LSTM deep learning model.")

# --- Sidebar for user inputs ---
st.sidebar.title("Parameters")
stock_tickers = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "NVDA", "JPM", "V", "PG"]
selected_ticker = st.sidebar.selectbox("Select a Stock Ticker:", stock_tickers)

look_back = st.sidebar.slider("Lookback Period (days):", 10, 60, 30)

if st.sidebar.button("Run Prediction"):
    st.info(f"Running prediction for {selected_ticker} with a lookback period of {look_back} days.")

    # --- Data Fetching and Preprocessing ---
    @st.cache_data
    def get_data(ticker):
        try:
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=365 * 4)
            df = yf.download(ticker, start=start_date, end=end_date)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    df = get_data(selected_ticker)

    if df is not None and not df.empty:
        df = df.reset_index()
        data = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Splitting data
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]
        test_data = scaled_data[training_data_len - look_back:, :]

        def create_dataset(dataset, look_back):
            X, y = [], []
            for i in range(look_back, len(dataset)):
                X.append(dataset[i-look_back:i, :])
                y.append(dataset[i, 3])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_data, look_back)
        X_test, y_test = create_dataset(test_data, look_back)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        # --- Building and Training the LSTM Model ---
        st.subheader("Building and Training the LSTM Model")
        with st.spinner("Training LSTM model..."):
            lstm_model = Sequential()
            lstm_model.add(LSTM(50, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(LSTM(50, return_sequences=False))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(Dense(25))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = lstm_model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.1, callbacks=[early_stopping], verbose=0)

        # Make predictions on test data
        lstm_predictions = lstm_model.predict(X_test)
        
        # --- Inverse Scaling and Evaluation ---
        st.subheader("Model Evaluation")
        y_test_unscaled = scaler.inverse_transform(np.c_[np.zeros((len(y_test), 5)), y_test])[:, -1]
        lstm_predictions_unscaled = scaler.inverse_transform(np.c_[np.zeros((len(lstm_predictions), 5)), lstm_predictions])[:, -1]

        lstm_rmse = np.sqrt(mean_squared_error(y_test_unscaled, lstm_predictions_unscaled))
        lstm_r2 = r2_score(y_test_unscaled, lstm_predictions_unscaled)

        st.success(f"LSTM Model RMSE: **{lstm_rmse:.4f}**")
        st.success(f"LSTM Model R-squared: **{lstm_r2:.4f}**")
        
        # --- Feature 1: Training Loss Graph ---
        st.subheader("Model Training History")
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

        # --- Feature 2: Stock Price Prediction Visualization ---
        st.subheader("Stock Price Prediction Visualization")
        trace_actual = go.Scatter(x=df['Date'][training_data_len + look_back:], y=y_test_unscaled, mode='lines', name='Actual Price')
        trace_predicted = go.Scatter(x=df['Date'][training_data_len + look_back:], y=lstm_predictions_unscaled, mode='lines', name='Predicted Price')
        fig = go.Figure(data=[trace_actual, trace_predicted])
        fig.update_layout(title=f"{selected_ticker} Price Prediction", xaxis_title="Date", yaxis_title="Stock Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Feature 3: Future Price Prediction (tabular and graph) ---
        st.subheader("Future Price Prediction for the Next Period")
        
        last_look_back_data = scaled_data[-look_back:]
        last_look_back_data = np.reshape(last_look_back_data, (1, look_back, scaled_data.shape[1]))
        
        # Make a single prediction for the next period
        next_period_prediction_scaled = lstm_model.predict(last_look_back_data)[0][0]
        
        # Inverse transform the prediction
        next_period_prediction_unscaled = scaler.inverse_transform(np.c_[np.zeros(5), next_period_prediction_scaled])[0][-1]
        
        # Display the prediction in a table
        future_date = df['Date'].iloc[-1] + datetime.timedelta(days=1)
        prediction_df = pd.DataFrame({
            "Date": [future_date.strftime('%Y-%m-%d')],
            "Predicted Price": [f"${next_period_prediction_unscaled:.2f}"]
        })
        st.dataframe(prediction_df, use_container_width=True)
        
        # Plot the future prediction on the graph
        last_date = df['Date'].iloc[-1]
        future_df = pd.DataFrame({
            'Date': [last_date, future_date],
            'Price': [df['Close'].iloc[-1], next_period_prediction_unscaled]
        })
        
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Price'))
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Price'], mode='lines', name='Future Prediction', line=dict(dash='dash', color='orange')))
        fig_future.update_layout(title=f"{selected_ticker} Historical and Future Prediction", xaxis_title="Date", yaxis_title="Stock Price")
        st.plotly_chart(fig_future, use_container_width=True)

    else:
        st.error("No data found for the selected ticker. Please try a different one.")
This video shows how to build a stock prediction web app in Python using Streamlit, Yahoo Finance, and Facebook Prophet. [Build A Stock Prediction Web App In Python](https://www.youtube.com/watch?v=0E_31WqVzCY)
http://googleusercontent.com/youtube_content/8
