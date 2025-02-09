import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def modified_alexnet(input_shape=(64, 64, 1), num_classes=2):
    model = Sequential([
        # Convolutional Layer 1
        Conv2D(96, kernel_size=(11, 11), strides=4, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # Convolutional Layer 2
        Conv2D(256, kernel_size=(5, 5), strides=1, activation='relu', padding="same"),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # Convolutional Layer 3
        Conv2D(384, kernel_size=(3, 3), strides=1, activation='relu', padding="same"),

        # Convolutional Layer 4
        Conv2D(384, kernel_size=(3, 3), strides=1, activation='relu', padding="same"),

        # Convolutional Layer 5
        Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding="same"),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # Flatten and Fully Connected Layers
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    model = modified_alexnet()
    model.summary()
