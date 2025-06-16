import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# === Parameters ===
time_steps = 20
forward_days = 6
classes = 5
max_samples = 30

# === Load filtered current data ===
df = pd.read_csv('goog_curr.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# === Ensure features exist (dummy fill if missing) ===
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'MA10', 'MA20', 'MA50', 'MACD_ml', 'MACD_his', 
            'RSI14', 'OBV', 'ATR', 'StochasticMomentum']
for feat in features:
    if feat not in df.columns:
        df[feat] = 0.0

# === Normalize ===
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# === Create labeled sequences ===
X, y, dates = [], [], []
for i in range(len(scaled_data) - time_steps - forward_days):
    X.append(scaled_data[i:i+time_steps])
    
    current_price = df['Close'].iloc[i+time_steps-1]
    future_price = df['Close'].iloc[i+time_steps+forward_days-1]
    pct_change = (future_price - current_price) / current_price * 100

    if pct_change <= -2:
        label = 0
    elif -2 < pct_change <= -0.5:
        label = 1
    elif -0.5 < pct_change < 0.5:
        label = 2
    elif 0.5 <= pct_change < 2:
        label = 3
    else:
        label = 4

    y.append(label)
    dates.append(df['Date'].iloc[i+time_steps-1])  # use last date of sequence

X, y, dates = np.array(X), np.array(y), np.array(dates)

# === Split train/test by index (80/20) ===
split_index = int(0.8 * len(X))
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]
dates_train, dates_test = dates[:split_index], dates[split_index:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# === Resample ONLY train set ===
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_train_resampled, y_train_resampled = BorderlineSMOTE(kind='borderline-2').fit_resample(X_train_flat, y_train)
X_train_resampled = X_train_resampled.reshape(-1, time_steps, len(features))

# === Build & Train Model ===
def Create_LSTM_Model(units=1000, drop_out=0.5, lr=5e-4):
    model = Sequential([
        LSTM(units, input_shape=(time_steps, len(features)), return_sequences=True),
        Dropout(drop_out),
        BatchNormalization(),
        LSTM(units),
        Dropout(drop_out),
        BatchNormalization(),
        Dense(units, activation='tanh'),
        Dropout(drop_out),
        BatchNormalization(),
        Dense(classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = Create_LSTM_Model()
early_stop = EarlyStopping(patience=300, restore_best_weights=True, monitor='val_loss')

history = model.fit(
    X_train_resampled, y_train_resampled, epochs=1000, batch_size=512,
    validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1
)

# === Evaluate ===
model.evaluate(X_test, y_test)

y_pred = np.argmax(model.predict(X_test), axis=1)
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)

# === Per-class accuracy ===
unique_labels = np.unique(y_test)
print("Per-class accuracy:")
for i in unique_labels:
    if i < conf_mat.shape[0] and i < conf_mat.shape[1]:
        correct = conf_mat[i, i]
        total = sum(conf_mat[i, :])
        acc = 100 * correct / total if total > 0 else 0
        print(f"Label {i} accuracy: {acc:.1f}%")

# === Random sample results (non-synthetic) ===
results = []
np.random.seed()
sample_indices = np.random.choice(len(X_test), size=min(max_samples, len(X_test)), replace=False)

for i in sample_indices:
    idx_in_df = df[df['Date'] == dates_test[i]].index[0]
    current_price = df['Close'].iloc[idx_in_df]
    future_price = df['Close'].iloc[idx_in_df + forward_days]
    pct_change = (future_price - current_price) / current_price * 100

    results.append({
        'Current_Date': df['Date'].iloc[idx_in_df],
        'Future_Date': df['Date'].iloc[idx_in_df + forward_days],
        'Current_Price': current_price,
        'Future_Price': future_price,
        'Predicted_Label': y_pred[i],
        'True_Label': y_test[i],
        'Actual_Pct_Change': pct_change
    })

results_df = pd.DataFrame(results)
results_df.to_excel('prediction_results_curr.xlsx', index=False)
print("\nSaved prediction results to 'prediction_results_curr.xlsx'")
