import numpy as np
from tensorflow.keras.layers import Input, Dense , Flatten
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt




num_class = 10
EPOCH = 10
BATCH_SIZE = 16

def built_model():
    #Model
    inputs = Input((28,28))
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)     
    outputs = Dense(num_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

def main():
    #Load data
    (trainX, trainY),(testX,testY) = load_data()

    print(trainX.shape, trainY.shape)
    print(testX.shape,testY.shape)

    #Normalization
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0


    trainY = to_categorical(trainY, num_class)
    testY = to_categorical(testY, num_class)


    #compile
    model = built_model()
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    history = model.fit(trainX,trainY,EPOCH,BATCH_SIZE, validation_data = (testX,testY))


# Plot Loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()