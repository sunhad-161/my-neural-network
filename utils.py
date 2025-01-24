import numpy as np

def load_dataset():
    with np.load("mnist.npz") as f:
        # RGB -> UnitRGB
        x_train = f['x_train'].astype("float32") / 255

        # изменение формы массива из (60000, 28, 28) в (60000, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

        # массив цифр [3, 1, 4...]
        y_train = f['y_train']

        # конвертируем в удобный формат [[0, 0, 0, 1, 0 .. 0], [0, 1, 0 .. 0], [0, 0, 0, 0, 1, 0 .. 0]...]
        y_train = np.eye(10)[y_train]

        return x_train, y_train
    
def load_neural_network():
    filepath = "neural_network.npz"
    data = np.load(filepath)
    return (
        data["weights_input_to_hidden"],
        data["weights_hidden_to_output"],
        data["bias_input_to_hidden"],
        data["bias_hidden_to_output"],
    )