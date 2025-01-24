import numpy as np
import matplotlib.pyplot as plt
import random

import utils

images, labels = utils.load_dataset()

# матрицы весов
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

# нейроны смещения
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

epochs = 3 # количество эпох
learning_rate = 0.01 # скорость обучения
e_loss = 0
e_correct = 0

for epoch in range(epochs):
	print(f"Epoch №{epoch}")

	for image, label in zip(images, labels):
		image = np.reshape(image, (-1, 1))
		label = np.reshape(label, (-1, 1))

		# прямой проход (от входных в скрытые)
		hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
		hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid

		# прямой проход (от скрытых к выходным)
		output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
		output = 1 / (1 + np.exp(-output_raw))
		
        # проверка
		e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
		e_correct += int(np.argmax(output) == np.argmax(label))

		# обратный проход (скрытый)
		delta_output = output - label
		weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
		bias_hidden_to_output += -learning_rate * delta_output

		# обратный проход (входной)
		delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
		weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
		bias_input_to_hidden += -learning_rate * delta_hidden
	
    # корректность работы нс после эпох
	print(f"loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
	print(f"correct: {round((e_correct / images.shape[0]) * 100, 3)}%")
	e_loss = 0
	e_correct = 0

test_image = random.choice(images)
image = np.reshape(test_image, (-1, 1))

hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw))

output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"Это цифра: {output.argmax()}")
plt.show()