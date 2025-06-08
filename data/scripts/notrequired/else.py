import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# === Parameters ===
time_steps = 20
forward_days = 6
max_samples = 30

# === Load Data ===
df = pd.read_csv('goog_curr.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# === Feature Setup ===
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'MA10', 'MA20', 'MA50', 'MACD_ml', 'MACD_his',
            'RSI14', 'OBV', 'ATR', 'StochasticMomentum']

for feat in features:
    if feat not in df.columns:
        df[feat] = 0.0

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# === Create Sequences for Regression ===
X, y, seq_dates = [], [], []
for i in range(len(df) - time_steps - forward_days):
    X.append(scaled_data[i:i+time_steps])
    current_close = df['Close'].iloc[i + time_steps - 1]
    future_close = df['Close'].iloc[i + time_steps + forward_days - 1]
    pct_change = (future_close - current_close) / current_close * 100
    
    # Clip pct_change to avoid extreme targets
    pct_change = np.clip(pct_change, -20, 20)  # adjust clipping based on your domain knowledge
    
    y.append(pct_change)
    seq_dates.append(df['Date'].iloc[i + time_steps - 1])

X = np.array(X)
y = np.array(y)
seq_dates = np.array(seq_dates)

# === Train/Test Split (by date) ===
train_cutoff = pd.Timestamp("2025-05-23")
is_train = seq_dates <= train_cutoff
is_test = seq_dates > train_cutoff

X_train, y_train = X[is_train], y[is_train]
X_test, y_test = X[is_test], y[is_test]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# === Scale target variable y ===
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1,1)).flatten()

# === Model ===
def create_model(units=512, lr=1e-4, dropout_rate=0.5):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(time_steps, len(features))),
        Dropout(dropout_rate),
        BatchNormalization(),
        LSTM(units),
        Dropout(dropout_rate),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)  # Regression output
    ])
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)  # gradient clipping added
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

model = create_model()

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=100, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=50, factor=0.5, verbose=1)
]

# === Training ===
history = model.fit(
    X_train, y_train_scaled,
    epochs=2000,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=2,
    shuffle=False
)

# === Check for NaN or inf in test data ===
print("Checking test data for NaN or inf values...")
print("y_test contains NaN?", np.isnan(y_test).any())
print("y_test contains inf?", np.isinf(y_test).any())
print("X_test contains NaN?", np.isnan(X_test).any())
print("X_test contains inf?", np.isinf(X_test).any())

# === Predict and evaluate ===
try:
    y_pred_scaled = model.predict(X_test)
    # Inverse scale predictions
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Evaluate manually since scaled predictions
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
except Exception as e:
    print("Error during evaluation:", e)

# === Predict Future (Next 2 Weeks) ===
last_sequence = scaled_data[-time_steps:]
pred_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=14, freq='B')  # business days

future_preds = []
current_sequence = last_sequence.copy()

for pred_date in pred_dates:
    x_input = current_sequence[-time_steps:].reshape(1, time_steps, len(features))
    pct_future_scaled = model.predict(x_input)[0][0]
    pct_future = y_scaler.inverse_transform([[pct_future_scaled]])[0][0]

    # Clip prediction to reasonable range to prevent crazy jumps
    pct_future = np.clip(pct_future, -20, 20)

    last_close = df['Close'].iloc[-1] if len(future_preds) == 0 else future_preds[-1]['Future_Close']
    future_close = last_close * (1 + pct_future / 100)

    future_preds.append({
        'Date': pred_date,
        'Predicted_Change(%)': pct_future,
        'Future_Close': future_close
    })

    # Update sequence: simulate new row features (only close changes realistically here)
    # For other features, you may want to model or copy from previous data
    new_row = current_sequence[-1].copy()
    close_idx = features.index('Close')
    # Transform future_close back to scaled feature space
    # Create a dummy row with zero except Close feature = future_close to scale properly
    dummy = np.zeros((1, len(features)))
    dummy[0, close_idx] = future_close
    scaled_close = scaler.transform(dummy)[0, close_idx]
    new_row[close_idx] = scaled_close
    current_sequence = np.vstack([current_sequence[1:], new_row])

# === Save Results ===
results_df = pd.DataFrame(future_preds)
results_df.to_excel('predicted_future_regression.xlsx', index=False)
print("Saved future predictions to 'predicted_future_regression.xlsx'")
