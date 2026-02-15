# ==============================
# HYBRID RF + RNN IDS SYSTEM
# ==============================

import pandas as pd
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ==============================
# 1. LOAD DATA
# ==============================

DATA_PATH = "data/"

files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

print("Found files:", files)

df_list = []

for file in files:
    print("Loading:", file)

    # Limit rows for safety (remove later if PC is strong)
    temp = pd.read_csv(file, nrows=100000)

    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print("Total rows:", len(df))
print(df.head())


# ==============================
# 2. CLEAN DATA
# ==============================

print("\nCleaning data...")

# Remove spaces in column names
df.columns = df.columns.str.strip()

# Replace infinite with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop missing values
df.dropna(inplace=True)

print("After cleaning:", df.shape)


# ==============================
# 3. PREPARE FEATURES & LABEL
# ==============================

print("\nPreparing features and labels...")

X = df.drop("Label", axis=1)
y = df["Label"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Classes:", encoder.classes_)


# ==============================
# 4. TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# ==============================
# 5. FEATURE SCALING
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# 6. RANDOM FOREST (BASELINE)
# ==============================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)

print("\n=== RANDOM FOREST RESULTS ===")
print(classification_report(y_test, rf_pred))


# ==============================
# 7. FEATURE SELECTION (RF)
# ==============================

print("\nSelecting important features...")

importances = rf.feature_importances_

# Select top features
N_FEATURES = 30   # You may change this

indices = np.argsort(importances)[-N_FEATURES:]

X_train_sel = X_train_scaled[:, indices]
X_test_sel = X_test_scaled[:, indices]

print("Selected features:", N_FEATURES)


# ==============================
# 8. RESHAPE FOR RNN
# ==============================

# RNN expects: (samples, timesteps, features)
# We use 1 timestep (simplified sequence)

X_train_rnn = X_train_sel.reshape(
    X_train_sel.shape[0], 1, X_train_sel.shape[1]
)

X_test_rnn = X_test_sel.reshape(
    X_test_sel.shape[0], 1, X_test_sel.shape[1]
)


# ==============================
# 9. BUILD RNN MODEL
# ==============================

print("\nBuilding RNN model...")

model = Sequential()

model.add(LSTM(
    64,
    input_shape=(1, N_FEATURES),
    return_sequences=False
))

model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))
model.add(Dense(len(np.unique(y)), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ==============================
# 10. TRAIN RNN
# ==============================

print("\nTraining RNN...")

history = model.fit(
    X_train_rnn,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)


# ==============================
# 11. EVALUATE HYBRID MODEL
# ==============================

print("\nEvaluating RNN...")

rnn_pred = model.predict(X_test_rnn)
rnn_pred_classes = np.argmax(rnn_pred, axis=1)

print("\n=== HYBRID RF + RNN RESULTS ===")
print(classification_report(y_test, rnn_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rnn_pred_classes))


# ==============================
# DONE
# ==============================

print("\nTraining complete.")
