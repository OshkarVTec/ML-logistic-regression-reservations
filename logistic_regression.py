# Oskar Adolfo Villa López
# Date: 23-08-2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

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


# Logistic regression
def h(params, bias, samples):
    # Compute the dot product of params and samples, then add the bias
    acum = np.dot(samples, params) + bias
    # Apply the sigmoid function
    acum = 1 / (1 + np.exp(-acum))
    return acum


def GD(params, bias, samples, y, alpha):
    # Compute the predictions for all samples
    predictions = h(params, bias, samples)

    # Compute the errors
    errors = predictions - y

    # Compute the gradient for the parameters
    gradient = np.dot(samples.T, errors) / len(samples)

    # Update the parameters
    params = params - alpha * gradient

    # Update the bias
    bias = bias - alpha * np.mean(errors)

    return params, bias


def compute_loss(params, bias, samples, y):
    # Compute the predictions for all samples
    predictions = h(params, bias, samples)

    # Compute the cross-entropy error
    epsilon = 1e-15  # To avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    error = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    return error


def train(params, samples, y, alpha=0.01, max_epochs=1000):
    errors = []
    epochs = 0
    bias = 0.0  # Initialize bias
    error = float("inf")  # Initialize error to a large value

    while error > 0.01 and epochs < max_epochs:
        params, bias = GD(params, bias, samples, y, alpha)
        error = compute_loss(params, bias, samples, y)
        errors.append(error)
        epochs += 1

    print("Training finished in", epochs, "epochs")
    return params, bias, errors


x_train_np = df_x_train_scaled.to_numpy()
y_train_np = y_train.to_numpy()
num_params = x_train_np.shape[1]  # Number of features
params = np.ones(num_params)
params, bias, errors = train(params, x_train_np, y_train_np, 0.1, 10000)


# Error plot
plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Training Error. lr = 0.1")
plt.show()


# Test
def test(params, bias, samples, y):

    predictions = np.array([h(params, bias, sample) for sample in samples])

    error = compute_loss(params, bias, samples, y)

    return predictions, error


y_test_np = y_test.to_numpy()
x_test_np = df_x_test_scaled.to_numpy()
test_predictions, test_error = test(params, bias, x_test_np, y_test_np)
train_predictions, train_error = test(params, bias, x_train_np, y_train_np)


# Metrics
def accuracy(predictions, y):
    # Convert predictions to binary (0 or 1) based on a threshold (e.g., 0.5)
    threshold = 0.5
    binary_predictions = (predictions >= threshold).astype(int)

    # Calculate accuracy
    correct_predictions = np.sum(binary_predictions == y)
    total_predictions = len(y)
    accuracy = correct_predictions / total_predictions
    return accuracy


threshold = 0.5
train_predictions_binary = (train_predictions >= threshold).astype(int)
test_predictions_binary = (test_predictions >= threshold).astype(int)

train_f1 = f1_score(y_train, train_predictions_binary)
test_f1 = f1_score(y_test, test_predictions_binary)

print("Train F1-score:", train_f1)
print("Test F1-score:", test_f1)
print("Train accuracy:", accuracy(train_predictions, y_train_np))
print("Test accuracy:", accuracy(test_predictions, y_test_np))

# Convert predictions to binary (0 or 1) based on a threshold (e.g., 0.5)
threshold = 0.5
binary_predictions = (test_predictions >= threshold).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_test_np, binary_predictions)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Test Confusion Matrix")
plt.show()

# Convert predictions to binary (0 or 1) based on a threshold (e.g., 0.5)
threshold = 0.5
binary_predictions = (train_predictions >= threshold).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_train_np, binary_predictions)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Train Confusion Matrix")
plt.show()
