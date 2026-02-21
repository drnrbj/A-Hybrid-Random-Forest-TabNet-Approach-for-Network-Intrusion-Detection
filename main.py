# ==========================================================
# HYBRID RANDOM FOREST + GRU IDS (CICIDS2017)
# Baseline: Random Forest
# Proposed: Hybrid RF + RNN
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
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# 1. LOAD DATA
# ==========================================================

DATA_PATH = "data/"
files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

df_list = []

for file in files:
    print("Loading:", file)
    temp = pd.read_csv(file, nrows=200000)
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print("Total rows:", len(df))

# ==========================================================
# 2. CLEAN DATA
# ==========================================================

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

num_classes = len(np.unique(y))
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

# ==========================================================
# 5. SCALING
# ==========================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 6. RANDOM FOREST (BASELINE MODEL)
# ==========================================================

print("\nTraining Random Forest (Baseline)...")

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)
rf_proba_full = rf.predict_proba(X_test_scaled)

print("\n=== RANDOM FOREST (BASELINE) RESULTS ===")
print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

# ==========================================================
# 7. FEATURE SELECTION USING RF
# ==========================================================

importances = rf.feature_importances_
N_FEATURES = 20

indices = np.argsort(importances)[-N_FEATURES:]

X_train_sel = X_train_scaled[:, indices]
X_test_sel = X_test_scaled[:, indices]

# ==========================================================
# 8. CREATE TEMPORAL SEQUENCES FOR RNN
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

gc.collect()

# ==========================================================
# 9. CLASS WEIGHTS FOR RNN
# ==========================================================

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_seq),
    y=y_train_seq
)

class_weight_dict = dict(zip(np.unique(y_train_seq), weights))

# ==========================================================
# 10. BUILD GRU MODEL (INTERNAL COMPONENT)
# ==========================================================

model = Sequential([
    GRU(128, return_sequences=True, input_shape=(SEQ_LENGTH, N_FEATURES)),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    metrics=["accuracy"]
)

# ==========================================================
# 11. TRAIN RNN (NO STANDALONE REPORT)
# ==========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

model.fit(
    X_train_seq,
    y_train_seq,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================================
# 12. HYBRID RF + RNN (WEIGHTED ENSEMBLE)
# ==========================================================

print("\nBuilding Hybrid RF + RNN Model...")

rnn_proba = model.predict(X_test_seq, verbose=0)

# Align RF probabilities to sequence size
rf_proba = rf_proba_full[SEQ_LENGTH:]

# Weighted soft voting
RF_WEIGHT = 0.90
RNN_WEIGHT = 0.10

ensemble_proba = (RF_WEIGHT * rf_proba) + (RNN_WEIGHT * rnn_proba)
ensemble_pred = np.argmax(ensemble_proba, axis=1)

print("\n=== HYBRID RF + RNN RESULTS ===")
print(classification_report(y_test_seq, ensemble_pred))
print(confusion_matrix(y_test_seq, ensemble_pred))

print("\nTraining Complete.")