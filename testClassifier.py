import cv2
import tensorflow as tf
import numpy as np
from SudokuParser import SudokuParser

'''
model = tf.keras.models.load_model('model.hdf5')

six = cv2.imread('one.jpg')
cv2.imshow('Image', six)
cv2.waitKey(0)
six_gray = cv2.cvtColor(six, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image',six_gray)
imgCrop = tf.constant([six_gray])
imgCrop = tf.expand_dims(imgCrop, axis = 3)
imgCrop = tf.image.resize(imgCrop, [28, 28])
print(np.argmax(model.predict(imgCrop)))
'''

sudokuParser = SudokuParser('sudoku.jpg', 'model.hdf5')
print(sudokuParser.get_matrix_from_image())