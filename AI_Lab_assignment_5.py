import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten,Conv2D, MaxPooling2D, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


(trainX, trainY), (testX, testY) = mnist.load_data()


trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# CNN expects (height, width, channels)
trainX = trainX[..., tf.newaxis]  
testX = testX[..., tf.newaxis]



#CNN Model Create
inputs = Input((28, 28, 1))  # include channel dimension for grayscale
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)


# Train Model

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY, validation_split=0.1, epochs=5, batch_size=32, verbose=1)




# Evaluate on test set
test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")


# Plot training history
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.savefig("training_accuracy.png")
plt.legend()
plt.show()


# Predict on test set
predictions = model.predict(testX)
predicted_labels = np.argmax(predictions, axis=1)

# Show 20 test images with predicted labels in a grid
plt.figure(figsize=(12, 8))

for i in range(10):
    plt.subplot(2, 5, i+1)  # 2 rows, 5 columns
    plt.imshow(testX[i], cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {testY[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("test_predictions.png")
plt.show()
