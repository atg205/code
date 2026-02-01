# train_xgboost.py

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

# --------------------
# Load data
# --------------------
X_train = np.load("X_train.npy",allow_pickle=True)
X_test = np.load("X_test.npy",allow_pickle=True)
y_train = np.load("y_train.npy",allow_pickle=True)
y_test = np.load("y_test.npy",allow_pickle=True)
label_classes = np.load("label_classes.npy", allow_pickle=True)

num_classes = len(label_classes)

# --------------------
# Model
# --------------------
model = xgb.XGBClassifier(
    objective="multi:softprob" if num_classes > 2 else "binary:logistic",
    num_class=num_classes if num_classes > 2 else None,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss" if num_classes > 2 else "logloss",
    random_state=42
)

# --------------------
# Train
# --------------------
model.fit(X_train, y_train)

# --------------------
# Evaluate
# --------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_classes))

# --------------------
# Save model
# --------------------
model.save_model("xgboost_model.json")
