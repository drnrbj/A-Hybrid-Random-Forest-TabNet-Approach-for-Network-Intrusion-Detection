# ==========================================================
# HYBRID RANDOM FOREST + LSTM IDS (CICIDS2017)
# ==========================================================

import pandas as pd
import numpy as np
import glob
import os
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# 1. LOAD DATA
# ==========================================================

DATA_PATH = "data/"
files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

print("Found files:", files)

df_list = []

for file in files:
    print("Loading:", file)
    temp = pd.read_csv(file, nrows=100000)  # adjust if needed
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print("Total rows:", len(df))


# ==========================================================
# 2. CLEAN DATA
# ==========================================================

print("\nCleaning data...")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("After cleaning:", df.shape)


# ==========================================================
# 3. PREPARE FEATURES & LABELS
# ==========================================================

X = df.drop("Label", axis=1)
y = df["Label"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Classes:", encoder.classes_)

# ==========================================================
# 4. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# ==========================================================
# 5. SCALING
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================================
# 6. RANDOM FOREST (BASE MODEL)
# ==========================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)

print("\n=== RANDOM FOREST RESULTS ===")
print(classification_report(y_test, rf_pred))


# ==========================================================
# 7. FEATURE SELECTION USING RF
# ==========================================================

print("\nSelecting important features...")

importances = rf.feature_importances_
N_FEATURES = 20

indices = np.argsort(importances)[-N_FEATURES:]

X_train_sel = X_train_scaled[:, indices]
X_test_sel = X_test_scaled[:, indices]

print("Selected top features:", N_FEATURES)


# ==========================================================
# 8. CREATE TEMPORAL SEQUENCES
# ==========================================================

def create_sequences(data, labels, seq_length=10):
    X_seq, y_seq = [], []
    
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(labels[i+seq_length])
    
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 10

X_train_seq, y_train_seq = create_sequences(X_train_sel, y_train, SEQ_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_sel, y_test, SEQ_LENGTH)

print("Sequence shape:", X_train_seq.shape)

gc.collect()


# ==========================================================
# 9. COMPUTE CLASS WEIGHTS
# ==========================================================

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_seq),
    y=y_train_seq
)

class_weight_dict = dict(enumerate(class_weights))


# ==========================================================
# 10. BUILD LSTM MODEL
# ==========================================================

print("\nBuilding LSTM model...")

model = Sequential()

model.add(LSTM(
    64,
    input_shape=(SEQ_LENGTH, N_FEATURES)
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


# ==========================================================
# 11. TRAIN LSTM
# ==========================================================

print("\nTraining LSTM...")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=25,
    batch_size=128,
    validation_split=0.1,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)


# ==========================================================
# 12. EVALUATE LSTM
# ==========================================================

print("\nEvaluating LSTM...")

rnn_proba = model.predict(X_test_seq)
rnn_pred = np.argmax(rnn_proba, axis=1)

print("\n=== LSTM RESULTS ===")
print(classification_report(y_test_seq, rnn_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_seq, rnn_pred))


# ==========================================================
# 13. HYBRID ENSEMBLE (RF + LSTM)
# ==========================================================

print("\nBuilding Ensemble...")

# Align RF predictions to sequence size
rf_proba = rf.predict_proba(X_test_scaled[SEQ_LENGTH:])

ensemble_proba = (rf_proba + rnn_proba) / 2
ensemble_pred = np.argmax(ensemble_proba, axis=1)

print("\n=== HYBRID RF + LSTM RESULTS ===")
print(classification_report(y_test_seq, ensemble_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_seq, ensemble_pred))

print("\nTraining Complete.")
