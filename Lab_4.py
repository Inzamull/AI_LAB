from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def my_polynomial(x):
    return 5 * x**2 + 10 * x - 2

def data_process(n=50, low=-1000, high=1000):
    x = np.random.randint(low, high, size=n).astype(np.float32)
    y = my_polynomial(x).astype(np.float32)
    return x, y

def prepare_train_val_test():
    x, y = data_process(n=1000, low=-1000, high=1000)

    n = len(x)
    idx = np.random.permutation(n)
    x, y = x[idx], y[idx]

    train_n = int(n * 0.7)
    val_n   = int(n * 0.1)
    test_n  = n - (train_n + val_n)  

    trainX = x[:train_n]
    trainY = y[:train_n]

    valX   = x[train_n : train_n + val_n]
    valY   = y[train_n : train_n + val_n]

    testX  = x[train_n + val_n :]
    testY  = y[train_n + val_n :]

    x_mean, x_std = trainX.mean(axis=0), trainX.std(axis=0) + 1e-8
    y_mean, y_std = trainY.mean(axis=0), trainY.std(axis=0) + 1e-8

    # Apply normalization
    trainX = (trainX - x_mean) / x_std
    valX   = (valX   - x_mean) / x_std
    testX  = (testX  - x_mean) / x_std

    trainY = (trainY - y_mean) / y_std
    valY   = (valY   - y_mean) / y_std
    testY  = (testY  - y_mean) / y_std

    print(f"n: {n}, train_n: {train_n}, val_n: {val_n}, test_n: {test_n}")
    return (trainX, trainY), (valX, valY), (testX, testY)

def build_model():
    inputs = Input((1, ))

    h1 = Dense(4,  activation='relu', name='hidden_layer_1')(inputs)
    h2 = Dense(8,  activation='relu', name='hidden_layer_2')(h1)
    h3 = Dense(16, activation='relu', name='hidden_layer_3')(h2)
    h4 = Dense(8,  activation='relu', name='hidden_layer_4')(h3)
    h5 = Dense(4,  activation='relu', name='hidden_layer_5')(h4)

    outputs = Dense(1, name='output_layer')(h5)
    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

def main():
    # Build & compile
    model = build_model()
    model.compile(optimizer='adam', loss='mse')

    # Data
    (trainX, trainY), (valX, valY), (testX, testY) = prepare_train_val_test()

    # Train
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=100)
    
    # Evaluate on TEST set (not on train)
    test_loss = model.evaluate(testX, testY, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")

    # Predict on TEST set
    y_pred = model.predict(testX, verbose=0)

    # R² ("accuracy") on TEST set
    r2 = r2_score(testY.ravel(), y_pred.ravel())
    print(f"R² score (test): {r2:.4f}  (~{r2*100:.2f}% accuracy)")

    plt.figure(figsize=(8,5))
    plt.scatter(testX, testY, label="Original f(x)", alpha=0.6)
    plt.scatter(testX, y_pred, label="Predicted f(x)", alpha=0.6)
    plt.xlabel("x (normalized)")
    plt.ylabel("f(x) (normalized)")
    plt.title("Original vs Predicted (Test Set)")
    plt.legend()
    plt.grid(True)
    #plt.savefig("comparison_plot.png")  # for LaTeX
    plt.show()

    

if __name__ == '__main__':
    main()