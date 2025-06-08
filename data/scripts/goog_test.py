import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# === Parameters ===
time_steps = 40
forward_days = 6
classes = 5
max_samples = 30  # Maximum number of random samples to include in Excel

# === Load data ===
df = pd.read_csv('GOOGL.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# === Calculate indicators ===
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_his'] = df['MACD'] - df['MACD_signal']

df.dropna(inplace=True)
df = df.reset_index(drop=True)

# === Feature selection ===
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'MA10', 'MA20', 'MA50', 'MACD_ml', 'MACD_his', 
            'RSI14', 'OBV', 'ATR', 'StochasticMomentum']

# Make sure these columns exist or calculate them beforehand.
# For now, let's check and create missing columns with zeros to avoid errors:
for feat in features:
    if feat not in df.columns:
        df[feat] = 0.0

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

# === Apply BorderlineSMOTE ONLY on X and y ===
X_flat = X.reshape(X.shape[0], -1)
X_resampled, y_resampled = BorderlineSMOTE(kind='borderline-2').fit_resample(X_flat, y)
X_resampled = X_resampled.reshape(-1, time_steps, len(features))

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

# === Define model ===
def Create_LSTM_Model(units=1000, drop_out=0.5, lr=5e-4):
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

# === Train model ===
model = Create_LSTM_Model()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(patience=300, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=1000, batch_size=512, validation_data=(X_test, y_test), callbacks=[early_stop])

# === Evaluate model ===
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
        total = sum(conf_mat[i, :])
        acc = 100 * correct / total if total > 0 else 0
        print(f"Label {i} accuracy: {acc:.1f}%")
    else:
        print(f"Label {i} not present in confusion matrix")

# === Map test samples back to original indices where possible ===
X_flat_original = X.reshape(X.shape[0], -1)

idx_test = []
for test_sample in X_test.reshape(X_test.shape[0], -1):
    matches = np.where((X_flat_original == test_sample).all(axis=1))[0]
    if len(matches) > 0:
        idx_test.append(matches[0])
    else:
        idx_test.append(-1)  # synthetic sample

idx_test = np.array(idx_test)

# === Randomly select indices for comparison ===
np.random.seed()  # Ensure randomness each run
non_synthetic_indices = np.where(idx_test != -1)[0]
random_indices = np.random.choice(non_synthetic_indices, size=min(max_samples, len(non_synthetic_indices)), replace=False)

# === Create a DataFrame with predictions and actual values for random dates ===
results = []
for i in random_indices:
    original_idx = idx_test[i]
    current_price = df.iloc[original_idx + time_steps - 1]['Close']
    future_price = df.iloc[original_idx + time_steps + forward_days - 1]['Close']
    actual_pct_change = (future_price - current_price) / current_price * 100
    predicted_label = y_pred[i]
    true_label = y_test[i]

    results.append({
        'Current_Date': df.iloc[original_idx + time_steps - 1]['Date'],
        'Future_Date': df.iloc[original_idx + time_steps + forward_days - 1]['Date'],
        'Current_Price': current_price,
        'Future_Price': future_price,
        'Predicted_Label': predicted_label,
        'True_Label': true_label,
        'Actual_Pct_Change': actual_pct_change
    })

results_df = pd.DataFrame(results)

# Save to Excel
results_df.to_excel('prediction_results.xlsx', index=False)

print("\nSaved prediction results to 'prediction_results.xlsx'")

# === Manual inspection example for one random test sample (non-synthetic) ===
if len(random_indices) > 0:
    sample_i = random_indices[0]
    original_df_index = idx_test[sample_i]

    print("\n=== Manual Prediction Inspection ===")
    print(f"Sample index in df: {original_df_index}")
    print("Current date:", df.iloc[original_df_index + time_steps - 1]['Date'])
    print("Future date:", df.iloc[original_df_index + time_steps + forward_days - 1]['Date'])
    print("Current price:", df.iloc[original_df_index + time_steps - 1]['Close'])
    print("Future price:", df.iloc[original_df_index + time_steps + forward_days - 1]['Close'])
    print(f"Actual % change: {((df.iloc[original_df_index + time_steps + forward_days - 1]['Close'] - df.iloc[original_df_index + time_steps - 1]['Close']) / df.iloc[original_df_index + time_steps - 1]['Close']) * 100:.2f}%")
    print(f"True label: {y_test[sample_i]} (actual movement class)")
    print(f"Predicted label: {y_pred[sample_i]}")
else:
    print("No non-synthetic test samples found for manual inspection.")