''' 
MNIST data, classification problem.
'''

#======================= Necessary Imports =========================
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical


#======================= Model Execution Flow =========================
def main():
    #--- Build model
    model = build_model()

    #--- Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #--- Load data
    (trainX, trainY), (testX, testY) = load_data()

    #--- Cross-check
    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)

    #--- Normalize and one-hot encode
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0
    trainY = to_categorical(trainY, num_classes=10)
    testY = to_categorical(testY, num_classes=10)

    #--- Train model
    history = model.fit(trainX, trainY,
                        validation_split=0.1,
                        epochs=10,
                        verbose=2)

    #--- Evaluate on test data
    test_loss, test_acc = model.evaluate(testX, testY)
    print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    #--- Plot training history
    plot_history(history)

    #--- Predict and visualize some samples
    predictions = model.predict(testX)
    random_idxs = np.random.choice(len(testX), size=10, replace=False)

    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(random_idxs):
        plt.subplot(2, 5, i+1)
        plt.imshow(testX[idx].reshape(28, 28), cmap="gray")

        pred_label = np.argmax(predictions[idx])
        true_label = np.argmax(testY[idx])

        color = "green" if pred_label == true_label else "red"
        plt.title(f"Pred: {pred_label}, True: {true_label}", color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


#======================= Visualization =========================
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


#======================= Model Construction =========================
def build_model():
    inputs = Input((28, 28), name='input_layer')
    x = Flatten(name='flatten')(inputs)
    x = Dense(64, activation='relu', name='dense_1')(x)
    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dense(64, activation='relu', name='dense_3')(x)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)

    model = Model(inputs, outputs, name='mnist_classifier')
    model.summary()
    return model


#======================= Entry Point =========================
if __name__ == '__main__':
    main()
