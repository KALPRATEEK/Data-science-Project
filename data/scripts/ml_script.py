import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import BorderlineSMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import datetime as dt

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


# === Parameters ===
time_steps = 40
forward_days = 5
classes = 5

df = pd.read_csv('PLTR.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Calculate EMA (Exponential Moving Averages)
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

# Calculate MACD and Signal line
df['MACD'] = df['EMA12'] - df['EMA26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Calculate MACD Histogram
df['MACD_his'] = df['MACD'] - df['MACD_signal']

# Remove any rows with NaN after EMA/MACD calculations
df.dropna(inplace=True)
df = df.reset_index(drop=True)

# Then continue with your feature scaling and so on...


# === Feature selection ===
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'MA10', 'MA20', 'MA50', 'MACD_ml', 'MACD_his', 
            'RSI14', 'OBV', 'ATR', 'StochasticMomentum']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# === Create supervised samples ===
X, y = [], []

for i in range(len(scaled_data) - time_steps - forward_days):
    X.append(scaled_data[i:i+time_steps])
    
    current_price = df['Close'].iloc[i+time_steps-1]
    future_price = df['Close'].iloc[i+time_steps+forward_days-1]
    pct_change = (future_price - current_price) / current_price * 100
    
    if pct_change <= -2:
        label = 0  # Big drop
    elif -2 < pct_change <= -0.5:
        label = 1  # Small drop
    elif -0.5 < pct_change < 0.5:
        label = 2  # Neutral
    elif 0.5 <= pct_change < 2:
        label = 3  # Small rise
    else:
        label = 4  # Big rise
        
    y.append(label)

X, y = np.array(X), np.array(y)

# === Optional: Apply BorderlineSMOTE ===
X_reshaped = X.reshape(X.shape[0], -1)
X_resampled, y_resampled = BorderlineSMOTE(kind='borderline-2').fit_resample(X_reshaped, y)
X_resampled = X_resampled.reshape(-1, time_steps, len(features))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

# === Model definition ===
def Create_LSTM_Model(units=1200, drop_out=0.5, lr=5e-4):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(time_steps, len(features)), return_sequences=True))
    model.add(Dropout(drop_out))
    model.add(BatchNormalization())
    model.add(LSTM(units=units))
    model.add(Dropout(drop_out))
    model.add(BatchNormalization())
    model.add(Dense(units=units, activation='tanh'))
    model.add(Dropout(drop_out))
    model.add(BatchNormalization())
    model.add(Dense(units=5, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Training ===
model = Create_LSTM_Model()
history = model.fit(X_train, y_train, epochs=1000, batch_size=1022, validation_data=(X_test, y_test))

# === Evaluation ===
model.evaluate(X_test, y_test)
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)


from collections import Counter

counts_origin = Counter(y_test)
unique_labels = np.unique(y_test)

print("Per-class accuracy:")
for i in unique_labels:
    if i < conf_mat.shape[0] and i < conf_mat.shape[1]:
        correct = conf_mat[i, i]
        total = sum(conf_mat[i, :])  # total true samples of class i
        acc = 100 * correct / total if total > 0 else 0
        print(f"Label {i} accuracy: {acc:.1f}%")
    else:
        print(f"Label {i} not present in confusion matrix")

# === Evaluation ===
model.evaluate(X_test, y_test)
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_mat = confusion_matrix(y_test, y_pred)


# === Manually inspect one test sample ===

# REBUILD the index map used when creating X and y
X_indices = []
for i in range(len(scaled_data) - time_steps - forward_days):
    X_indices.append(i)
X_indices = np.array(X_indices)

# Re-split the indices just like the features
_, X_test_indices = train_test_split(X_indices, test_size=0.1, random_state=42)

# Choose a test sample index to inspect (between 0 and len(X_test)-1)
test_sample_idx = 5  # Change this number as desired

# Get original index in the full df for this test sample
original_df_index = X_test_indices[test_sample_idx]

# Display relevant info
print("\n=== Manual Prediction Inspection ===")
print(f"Sample index in df: {original_df_index}")
print("Current date:", df.iloc[original_df_index + time_steps - 1]['Date'])
print("Future date:", df.iloc[original_df_index + time_steps + forward_days - 1]['Date'])
print("Current price:", df.iloc[original_df_index + time_steps - 1]['Close'])
print("Future price:", df.iloc[original_df_index + time_steps + forward_days - 1]['Close'])
print(f"Actual % change: {((df.iloc[original_df_index + time_steps + forward_days - 1]['Close'] - df.iloc[original_df_index + time_steps - 1]['Close']) / df.iloc[original_df_index + time_steps - 1]['Close']) * 100:.2f}%")
print(f"True label: {y_test[test_sample_idx]} (actual movement class)")
print(f"Predicted label: {y_pred[test_sample_idx]}")
