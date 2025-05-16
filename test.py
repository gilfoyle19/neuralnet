import numpy as np
from nn.layer import Dense
from nn.activation import ReLU, linear
from nn.loss import MeanSquaredError
from nn.model import NeuralNet
from nn.optimizer import SGD
from nn.callback import CSVLogger

def generate_sine_wave_data(num_samples=1000):
    X = np.linspace(0, 2 * np.pi, num_samples).reshape(-1, 1)  # shape (num_samples, 1)
    y = np.sin(X)  # shape (num_samples, 1)
    return X, y

def main():
    # Generate dataset
    X, y = generate_sine_wave_data(num_samples=1000)

    # Split into train and test
    split = int(0.8 * X.shape[0])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = NeuralNet(
        layers=(
            (Dense(64), ReLU()),
            (Dense(64), ReLU()),
            (Dense(1), linear()),
        ),
        loss=MeanSquaredError(),
        optimizer=SGD(learning_rate=0.01),
    )

    # Setup CSVLogger callback
    callback = CSVLogger("sine_training_loss.csv", overwrite=True)

    # Train the model
    model.fit(X_train, y_train, epochs=100, callbacks=[callback], verbose=True)

    # Evaluate on test set
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.5f}")


if __name__ == "__main__":
    main()