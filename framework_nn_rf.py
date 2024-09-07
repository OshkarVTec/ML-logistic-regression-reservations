# Oskar Adolfo Villa López
# Date: 05-09-2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Load the data
df = pd.read_csv("Hotel Reservations.csv")


# Take a sample of the data
df_sample = df.sample(frac=1)
df_x = df_sample.drop(
    columns=[
        "booking_status",
        "Booking_ID",
        "arrival_year",
        "arrival_month",
        "arrival_date",
    ]
)  # Features
df_y = df_sample["booking_status"]  # Label

# One hot encoding
df_x_one_hot = pd.get_dummies(
    df_x, columns=["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
)
df_x_num = df_x_one_hot.copy()
for column in df_x_one_hot.columns:
    if df_x_one_hot[column].dtype == bool:
        df_x_num[column] = df_x_one_hot[column].astype(int)

# Split the data with stratification
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df_x_num, df_y, train_size=0.8, stratify=df_y, random_state=42
)


# Label encoding
def label_encoder(series):
    unique_values = series.unique()
    mapping = {value: index for index, value in enumerate(unique_values)}
    return series.map(mapping)


y_train = label_encoder(df_y_train)
y_test = label_encoder(df_y_test)


# Replace outliers
def replace_outliers_with_iqr(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    if upper_bound == 0:
        upper_bound = 1  # To account for the case where the IQR is 0
    df.loc[:, column_name] = df[column_name].clip(lower_bound, upper_bound)
    return df


df_x_train = df_x_train.copy()
df_x_train = replace_outliers_with_iqr(df_x_train, "no_of_week_nights")
df_x_train = replace_outliers_with_iqr(df_x_train, "lead_time")
df_x_train = replace_outliers_with_iqr(df_x_train, "no_of_previous_cancellations")
df_x_train = replace_outliers_with_iqr(
    df_x_train, "no_of_previous_bookings_not_canceled"
)


# Scaling
def min_max_scaler(df):
    scaled_df = df.copy()

    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        range_val = max_val - min_val

        if range_val == 0:
            scaled_df[column] = (
                0  # or scaled_df[column] = df[column] if you want to keep the original values
            )
        else:
            scaled_df[column] = (df[column] - min_val) / range_val

    return scaled_df


df_x_train_scaled = min_max_scaler(df_x_train)
df_x_test_scaled = min_max_scaler(df_x_test)


## Neural Network

# Check if a GPU is available
if tf.config.list_physical_devices("GPU"):
    device_name = tf.test.gpu_device_name()
    print("GPU found: {}".format(device_name))
else:
    device_name = "/device:CPU:0"
    print("No GPU found, using CPU.")

# Use the device for TensorFlow computations
with tf.device(device_name):
    # Model number 2
    model_2 = tf.keras.Sequential(
        [
            Dense(64, activation="relu", input_shape=(df_x_train_scaled.shape[1],)),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dense(32, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    # Model number 3 (Model 2 with regularization)
    model_3 = tf.keras.Sequential(
        [
            Dense(64, activation="relu", input_shape=(df_x_train_scaled.shape[1],)),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model_3.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model_3.fit(
        df_x_train_scaled, y_train, epochs=400, validation_split=0.15, batch_size=32
    )

# Get predictions on the test set
y_pred_prob = model_3.predict(df_x_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

try:
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
except KeyError:

    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
plt.title("Accuracy vs. epochs")
plt.ylabel("Acc")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="lower right")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss vs. epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")
plt.show()

# Predict on the test set
y_pred_test = (model_3.predict(df_x_test_scaled) > 0.5).astype(int)
y_pred_test = y_pred_test.flatten()

# Predict on the validation set
val_x = df_x_train_scaled.sample(frac=0.15, random_state=42)
val_y = y_train[val_x.index]
y_pred_val = (model_3.predict(val_x) > 0.5).astype(int)
y_pred_val = y_pred_val.flatten()

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Test Confusion Matrix")
plt.show()

# Calculate confusion matrix
cm = confusion_matrix(val_y, y_pred_val)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Validation Confusion Matrix")
plt.show()

# Calculate accuracies
test_accuracy = accuracy_score(y_pred_test, y_test)
val_accuracy = accuracy_score(y_pred_val, val_y)

# Plot accuracies
plt.bar(["Test Accuracy"], [test_accuracy])
plt.bar(["Validation Accuracy"], [val_accuracy])
plt.ylabel("Accuracy")
plt.title("Model Accuracies")
plt.show()


## Random Forest


# Hyper parameter max depth
rf = RandomForestClassifier(max_depth=10)
rf.fit(df_x_train_scaled, y_train)

# Predict on the test set
y_pred_test_rf = rf.predict(df_x_test_scaled)

# Calculate F1-score
f1 = f1_score(y_test, y_pred_test_rf)
print("F1-score:", f1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_test_rf)
print("Accuracy:", accuracy)

# Predict on the training set
y_pred_train_rf = rf.predict(df_x_train_scaled)

# Calculate confusion matrix for test set
cm_test = confusion_matrix(y_test, y_pred_test_rf)

# Plot confusion matrix for test set as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Test Confusion Matrix (Random Forest)")
plt.show()

# Calculate confusion matrix for train set
cm_train = confusion_matrix(y_train, y_pred_train_rf)

# Plot confusion matrix for train set as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Train Confusion Matrix (Random Forest)")
plt.show()
