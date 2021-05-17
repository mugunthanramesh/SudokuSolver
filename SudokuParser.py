import cv2
import numpy as np
import operator
import tensorflow as tf 

class SudokuParser:
	def __init__(self, img, digitsDetector):
		self.path = img
		self.detectorPath = digitsDetector
		self.grid = cv2.imread(img)
		self.gridGray = cv2.cvtColor(self.grid, cv2.COLOR_BGR2GRAY)
		self.model = tf.keras.models.load_model(digitsDetector)

	def preprocess(self):
		proc = cv2.GaussianBlur(self.gridGray.copy(), (9, 9), 0)
		proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		proc = cv2.bitwise_not(proc, proc)  
		kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
		proc = cv2.dilate(proc, kernel)
		return proc

	def find_edges(self, preprocessedImage):
		contours,h = cv2.findContours(preprocessedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		polygon = contours[0]
		bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
		top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
		bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
		top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
		#image = cv2.circle(imgC.copy(), tuple(polygon[top_right][0]), radius=5, color=(0, 255, 0), thickness=-1)
		#image = cv2.circle(image, tuple(polygon[top_left][0]), radius=5, color=(0, 255, 0), thickness=-1)
		#image = cv2.circle(image, tuple(polygon[bottom_left][0]), radius=5, color=(0, 255, 0), thickness=-1)
		#image = cv2.circle(image, tuple(polygon[bottom_right][0]), radius=5, color=(0, 255, 0), thickness=-1)
		top_left, top_right, bottom_right, bottom_left = polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]
		edges = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
		return edges

	def transformImage(self, edges):
		top_left, top_right, bottom_right, bottom_left = edges
		side = max([self.getDistance(bottom_right, top_right),self.getDistance(top_left, bottom_left), self.getDistance(bottom_right, bottom_left),self.getDistance(top_left, top_right) ])
		dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
		m = cv2.getPerspectiveTransform(edges, dst)
		img = cv2.warpPerspective(self.gridGray.copy(), m, (int(side), int(side)))
		img = cv2.bitwise_not(img, img)
		return img

	def extract_number_image(self, points):
		imgCrop = self.transform[int(points[0][1]):int(points[1][1]), int(points[0][0]):int(points[1][0])].copy()
		#cv2.imshow('Image', imgCrop)
		imgCrop = tf.constant([imgCrop])
		imgCrop = tf.expand_dims(imgCrop, axis = 3)
		imgCrop = tf.image.resize(imgCrop, [28, 28])
		return np.argmax(self.model.predict(imgCrop))

	def getDistance(self,a, b):
		x = abs(b[0] - a[0])
		y = abs(b[1] - a[1])
		return np.sqrt((x ** 2) + (y ** 2))

	def get_matrix_from_image(self):
		matrix = np.zeros((9,9))
		processedImage = self.preprocess()
		edges = self.find_edges(processedImage)
		transformedImage = self.transformImage(edges)

		self.transform = transformedImage

		side = transformedImage.shape[:1]
		side = side[0]/9

		for j in range(9):
			for i in range(9):
				a = (i * side, j * side)
				b = ((i + 1) * side, (j + 1) * side)
				matrix[j][i] = self.extract_number_image((a,b))
		return matrix