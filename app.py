import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set the page to a dark theme and wide layout
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for dark theme and styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #111;
        color: #eee;
    }
    .stApp {
        background-color: #111;
        color: #eee;
    }
    .main .block-container {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .sidebar .sidebar-content {
        background-color: #222;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSlider label, .stSelectbox label {
        color: #ddd;
    }
</style>
""", unsafe_allow_html=True)

st.title("LSTM Model for Stock Price Prediction 📈")
st.markdown("This application predicts stock prices using a standalone LSTM deep learning model.")

# --- Sidebar for user inputs (now with a selectbox for tickers) ---
st.sidebar.title("Stock Settings")
stock_tickers = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "NVDA", "JPM", "V", "PG"]
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", stock_tickers)
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365 * 4))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
look_back = st.sidebar.slider("Lookback Window:", 10, 120, 60)
epochs = st.sidebar.slider("Epochs:", 10, 200, 20)
batch_size = st.sidebar.slider("Batch Size:", 16, 128, 32)
future_days = st.sidebar.slider("Future Days:", 1, 30, 10)

if st.sidebar.button("Run Prediction"):
    st.info(f"Running prediction for {selected_ticker} with a lookback period of {look_back} days.")
    
    # --- Data Fetching and Preprocessing ---
    @st.cache_data
    def get_data(ticker, start_date, end_date):
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    df = get_data(selected_ticker, start_date, end_date)

    if df is not None and not df.empty:
        df = df.reset_index()
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if not available_columns or len(available_columns) < 5:
            st.error("Error: The fetched data is incomplete. Missing required columns for prediction.")
            st.warning(f"Available columns: {df.columns.tolist()}")
            st.stop()

        # Check if there is enough data for the lookback period
        if len(df) < look_back:
            st.error(f"Error: Not enough data for the lookback period of {look_back} days. Please select an earlier start date or a smaller lookback window.")
            st.stop()
        
        data = df[available_columns].values

        # --- Display Stock Data in a Table ---
        st.subheader(f"Stock Data for {selected_ticker}")
        st.dataframe(df[['Date'] + available_columns], use_container_width=True)
        
        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Splitting data
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]
        
        def create_dataset(dataset, look_back):
            X, y = [], []
            if len(dataset) < look_back:
                return np.array(X), np.array(y)
            for i in range(look_back, len(dataset)):
                X.append(dataset[i-look_back:i, :])
                y.append(dataset[i, available_columns.index('Close')])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_data, look_back)
        
        # Check if X_train is empty after creation
        if len(X_train) == 0:
            st.error("Error: Not enough data to create training samples. Please adjust your 'Start Date' or 'Lookback Window'.")
            st.stop()

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        # --- Building and Training the LSTM Model ---
        st.subheader("Building and Training the LSTM Model")
        with st.spinner("Training LSTM model..."):
            lstm_model = Sequential()
            lstm_model.add(LSTM(100, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
            lstm_model.add(Dropout(0.3))
            lstm_model.add(LSTM(100, return_sequences=False))
            lstm_model.add(Dropout(0.3))
            lstm_model.add(Dense(50))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            history = lstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early_stopping], verbose=0)
            
        # --- Training Loss Graph ---
        st.subheader("Model Training History")
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

        # --- Future Price Prediction ---
        st.subheader("Next Predictions")
        
        future_predictions = []
        last_look_back_data = scaled_data[-look_back:]

        for _ in range(future_days):
            last_look_back_data = np.reshape(last_look_back_data, (1, look_back, scaled_data.shape[1]))
            next_period_prediction_scaled = lstm_model.predict(last_look_back_data, verbose=0)[0][0]
            temp_array_future_prediction = np.zeros((1, len(available_columns)))
            temp_array_future_prediction[:, available_columns.index('Close')] = next_period_prediction_scaled
            next_period_prediction_unscaled = scaler.inverse_transform(temp_array_future_prediction)[0][available_columns.index('Close')]
            
            future_predictions.append(next_period_prediction_unscaled)
            
            new_row_scaled = np.array(temp_array_future_prediction[0])
            last_look_back_data = np.vstack([last_look_back_data[0][1:], new_row_scaled])

        # Create a dataframe for future predictions
        future_dates = [df['Date'].iloc[-1] + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": future_predictions
        })
        st.dataframe(prediction_df, use_container_width=True)

        # --- Historical Price Graph (from image) ---
        st.subheader("Historical Stock Price")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Close'))
        fig_hist.update_layout(title=f"{selected_ticker} Historical Close Price", xaxis_title="Date", yaxis_title="Stock Price", template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- Future Prediction Graph (from image) ---
        st.subheader("Historical and Future Price Prediction")
        
        # Create a continuous date range for the plot
        full_date_range = pd.date_range(start=df['Date'].iloc[0], end=prediction_df['Date'].iloc[-1])
        full_df = pd.DataFrame(index=full_date_range)
        
        # Merge historical data
        full_df = full_df.merge(df.set_index('Date')['Close'], how='left', left_index=True, right_index=True)
        full_df = full_df.rename(columns={'Close': 'Historical_Close'})

        # Merge future predictions
        full_df = full_df.merge(prediction_df.set_index('Date')['Predicted_Close'], how='left', left_index=True, right_index=True)
        
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=full_df.index, y=full_df['Historical_Close'], mode='lines', name='Historical Close'))
        fig_future.add_trace(go.Scatter(x=full_df.index, y=full_df['Predicted_Close'], mode='lines', name='Predicted Future', line=dict(dash='dash', color='orange')))
        
        fig_future.update_layout(title=f"{selected_ticker} Historical and Future Prediction", xaxis_title="Date", yaxis_title="Stock Price", template="plotly_dark")
        st.plotly_chart(fig_future, use_container_width=True)

    else:
        st.error("No data found for the selected ticker. Please try a different one.")
