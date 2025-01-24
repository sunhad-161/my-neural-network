import tkinter as tk
from tkinter import Canvas
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw

import utils


# Загрузка нейронной сети
(
    weights_input_to_hidden,
    weights_hidden_to_output,
    bias_input_to_hidden,
    bias_hidden_to_output,
) = utils.load_neural_network()
print("Нейронная сеть успешно загружена.")

# Функция для распознавания цифры
def predict_digit(image_array):
    image = np.reshape(image_array, (-1, 1))
    
    # Прямой проход (от входных в скрытые)
    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

    # Прямой проход (от скрытых к выходным)
    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    print(output)

    return output.argmax()

# Класс для графического интерфейса
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание цифр")
        
        # Создание холста для рисования
        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, pady=10, padx=10)
        
        # Кнопки управления
        self.predict_button = tk.Button(root, text="Распознать", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=5)
        
        self.clear_button = tk.Button(root, text="Очистить", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, pady=5)
        
        # Инициализация для рисования
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        
    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black", outline="black")
        self.draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="black")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
    
    def predict(self):
        # Преобразование изображения в формат 28x28
        resized_image = self.image.resize((28, 28)).convert("L")
        image_array = 255 - np.array(resized_image, dtype=np.float32)

        # Нормализация в диапазон [0, 1]
        image_array /= 255.0
        flattened_image = image_array.flatten()

        prediction = predict_digit(flattened_image)
        tk.messagebox.showinfo("Результат", f"Это цифра: {prediction}")


# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
