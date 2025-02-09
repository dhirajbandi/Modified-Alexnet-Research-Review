import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set dataset path
DATASET_PATH = "hidden personal content"
IMAGE_SIZE = (64, 64)

def load_data():
    images, labels = [], []
    for label, category in enumerate(["benign", "malignant"]):
        folder_path = os.path.join(DATASET_PATH, category)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            image = cv2.resize(image, IMAGE_SIZE)  # Resize to 64x64
            images.append(image)
            labels.append(label)

    images = np.array(images).reshape(-1, 64, 64, 1) / 255.0  # Normalize
    labels = np.array(labels)

    return train_test_split(images, labels, test_size=0.3, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
