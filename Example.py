# Импорт необходимых библиотек
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras 
from tensorflow import nn
from tensorflow import convert_to_tensor

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog


# Создание класса, для упрощения работы с импортируемой моделью 
class CNN:
    """Class simplifes model usage
    """
    def __init__(self, path=None):
        self.class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        if path is None:
            self.model = keras.models.load_model(os.getcwd()+r'\Dropout_model_old.h5')
        else:
            self.model = keras.models.load_model(path)
        
    def get_class_name(self, predictions):
        """Takes prediction array and returns class name

        Args:
            predictions (float array): Model predictions

        Returns:
            string: Class name
        """
        return self.class_names[np.argmax(predictions)]
    
    def predict_path(self, path):
        """Takes path to an image and returns prediction array

        Args:
            path (string): Image path

        Returns:
            ndarray: Prediction array
        """
        im = cv2.imread(path)
        im = cv2.resize(im, (176, 176), interpolation = cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.array([im])
        im = convert_to_tensor(im)
        return self.model.predict(im)


if __name__ == "__main__":
    # Выбор анализируемого изображения
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    
    # Отображение выбранного изображения
    img = cv2.imread(path)
    cv2.imshow("imported image", img)
    
    # Создание класса
    model = CNN()
    # Получение вероятностей принадлежности к классу
    pred = model.predict_path(path)

    lables = model.class_names
    sizes = (np.array(nn.softmax(pred[0]), dtype=np.float32))*100
    
    # Вывод круговой диограммы
    plt.pie(sizes, labels=lables, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()
    
    cv2.waitKey()