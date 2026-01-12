import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


@st.cache_resource
def load_assets():
    checkpoint = torch.load(
        'full_pipeline.pth', 
        map_location=torch.device('cpu'),
        weights_only=False 
    )
    
    config = checkpoint['config']
    
    model = PredictionModel(
        input_dim=config['input_dim'], 
        hidden_dim=config['hidden_dim'], 
        num_layers=config['num_layers'], 
        output_dim=config['output_dim']
    )
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval() 
    
    scaler = checkpoint['scaler']
    
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Error: 'full_pipeline.pth' not found. Please make sure the file is in the same directory as this script.")
    st.stop()

st.title("📈 Stock Price Predictor (LSTM)")
st.caption("Powered by PyTorch & Streamlit")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-07"))

if st.button("Generate Prediction"):
    with st.spinner(f"Fetching data for {ticker}..."):
        df = yf.download(ticker, start=start_date)
        
        if df.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
        else:
            try:
                data = df[['Close']].values
                
                scaled_data = scaler.transform(data)
                
                seq_length = 30
                X_input = []
                
                if len(scaled_data) <= seq_length:
                    st.warning(f"Not enough data points. Need at least {seq_length + 1} days of data.")
                else:
                    for i in range(len(scaled_data) - seq_length):
                        X_input.append(scaled_data[i : i + seq_length])
                    
                    X_input = np.array(X_input)
                    
                    X_tensor = torch.tensor(X_input, dtype=torch.float32)
                    
                    with torch.no_grad():
                        y_pred_scaled = model(X_tensor)
                    

                    y_pred = scaler.inverse_transform(y_pred_scaled.numpy())
                    

                    actual_prices = data[seq_length:]
                    
                    plot_dates = df.index[seq_length:]

                    st.subheader(f"Price Prediction vs Actual: {ticker}")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(plot_dates, actual_prices, label='Actual Price', color='blue', alpha=0.6)
                    ax.plot(plot_dates, y_pred, label='Predicted Price', color='red', alpha=0.8)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price (USD)")
                    ax.legend()
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    st.pyplot(fig)
                    
                    latest_actual = actual_prices[-1][0]
                    latest_pred = y_pred[-1][0]
                    error = latest_pred - latest_actual
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Latest Actual Price", f"${latest_actual:.2f}")
                    col2.metric("Model Prediction", f"${latest_pred:.2f}")
                    col3.metric("Difference", f"${error:.2f}", delta_color="inverse")
            
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
