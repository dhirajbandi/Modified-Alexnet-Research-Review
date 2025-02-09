import numpy as np
import tensorflow as tf
from model_alexnet import modified_alexnet
from data_preprocessing import load_data

# Load dataset
X_train, X_test, y_train, y_test = load_data()

# Load model
model = modified_alexnet()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("breast_cancer_model.h5")

# Save training history
np.save("training_history.npy", history.history)
