import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import joblib

# Load the dataset
dataset = pd.read_csv('solar_angle_dataset.csv')

# Display the first few rows
print(dataset.head())

# Preprocessing
def preprocess_data(df):
    # Handle missing values (if any)
    df = df.dropna()

    # Convert 'Date' column to datetime format while handling UTC offset
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Extract date and time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['Minute'] = df['Date'].dt.minute

    # Drop original 'Date' column
    df = df.drop(columns=['Date'])

    # Convert Latitude & Longitude to numeric
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # Separate features and target
    X = df.drop(columns=['Solar Zenith Angle'])
    y = df['Solar Zenith Angle']

    # Split into train & test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize/Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler for later use in backend
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")

    return X_train, X_test, y_train, y_test, scaler

# Preprocess the dataset
X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset)

# Define an improved deep learning model
def create_improved_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dense(1)  # Output layer (regression)
    ])
    
    # Compile with an adaptive learning rate
    optimizer = Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Create the improved model
input_shape = X_train.shape[1]
model = create_improved_model(input_shape)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # Increased epochs for better learning
    batch_size=32,
    verbose=1
)

# Evaluate the model
y_pred = model.predict(X_test)

# Compute performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.show()

# Plot training and validation metrics
plot_history(history)

# Save the model
model.save('solar_angle_improved_model.h5')
print("Model saved as 'solar_angle_improved_model.h5'")
