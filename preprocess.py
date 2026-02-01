import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------
# Config
# --------------------
CSV_PATH = "/home/atg205/Malware/iotpreprocessed/iot23_combined_new.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 2

# --------------------
# Load CSV
# --------------------
df = pd.read_csv(CSV_PATH)

# --------------------
# Remove duplicate rows
# --------------------
df = df.drop_duplicates().reset_index(drop=True)

# --------------------
# Remove benign traffic
# --------------------
df = df[df["label"] != "Benign"].reset_index(drop=True)

# --------------------
# Remove rare classes
# --------------------
label_counts = df["label"].value_counts()
valid_labels = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index
df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

# --------------------
# Encode labels (target)
# --------------------
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(df["label"])
np.save("label_classes.npy", y_encoder.classes_)

# --------------------
# Drop label + identifiers
# --------------------
drop_cols = [
    "label",
    "uid",
    "id.orig_h",
    "id.resp_h"
]

X = df.drop(columns=drop_cols, errors="ignore")

# --------------------
# Handle missing values
# --------------------
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("unknown")
    else:
        X[col] = X[col].fillna(0)

# --------------------
# Encode categorical features (NO one-hot)
# --------------------
categorical_cols = X.select_dtypes(include=["object"]).columns

feature_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    feature_encoders[col] = le

# Optional: save encoders if you need inference later
np.save("feature_encoders.npy", feature_encoders, allow_pickle=True)

# --------------------
# Convert to numpy
# --------------------
X = X.values.astype(np.float32)

# --------------------
# Train / test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# --------------------
# Save outputs
# --------------------
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Preprocessing complete.")
print(f"Samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(y_encoder.classes_)}")
