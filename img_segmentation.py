import numpy as np
from os import path
from PIL import Image

from mlp import MultiLayerPerceptron as MLP

def calculate_neighbours(matrix_position, matrix_size, shape='cross', radius=1):
	'''
	This function will return a vector with the position of the matrix_position's
	neighbours.

	Parameters
	----------
	matrix_position: matrix_position from the map's weights
	matrix_size: size of the matrix used
	radius: radius of the neighbourhood
	shape: shape of the neighbourhood used
		values
		------
		cross: neighbours from the top, bottom, left & right sides in the selected radius
		square: neighbours from the left and bottom sides, the result is a square neighbordhood
	'''
	matrix_row, matrix_col = matrix_position
	max_row, max_col = matrix_size

	neighbours_matrix = []

	if shape == 'cross':
		#Calculate neighbours in the radius
		for i in xrange(1, radius+1):
			#Up
			if (matrix_row) - i >= 0:
				neighbours_matrix.append((matrix_col, matrix_row-i))
			#Down
			if (matrix_row) + i < max_row:
				neighbours_matrix.append((matrix_col, matrix_row+i))
			#Left
			if (matrix_col) - i >= 0:
				neighbours_matrix.append((matrix_col-i, matrix_row))
			#Right
			if (matrix_col) + i < max_col:
				neighbours_matrix.append((matrix_col+i, matrix_row))
	elif shape == 'square':
		pass
		'''
		for row in range(radius):
			for col in range(radius):
				if (col < matrix_col):
					neighbours_matrix.append((matrix_col+col, matrix_row))
		'''
				
	return neighbours_matrix

def generate_trainig_set(path_img1, path_img2, shape='cross', radius=1):
	'''
	Function to create a training set from two images

	Parameters
	----------
	path_img1: path to the image from class 1
	path_img2: path to the image from class 0
	shape: shape that will be used in the neighbourhood, 'cross' as default
	radius: neighbourhood's block radius, 1 as default
	'''

	training_set = []

	# Load both images in gray sacale
	img_1 = Image.open(path_img1).convert('L')
	img_2 = Image.open(path_img2).convert('L')

	# Extract pixels maps
	pix_img_1 = img_1.load()
	pix_img_2 = img_2.load()
	
	# Get images' size
	img_1_cols, img_1_rows  =  img_1.size
	img_2_cols, img_2_rows  =  img_2.size

	# Create training set with the neighbours' average for each pixel
	for matrix_row in range(img_1_rows):
		for matrix_col in range(img_1_cols):
			neighbours = calculate_neighbours((matrix_row, matrix_col), 
											  (img_1_rows, img_1_cols), 
											  shape,
											  radius)

			neighbourhood_average = 0
			for neighbour in neighbours:
				neighbourhood_average += pix_img_1[neighbour[0], neighbour[1]]

			training_set.append((np.array(neighbourhood_average / len(neighbours)), 
								np.array(1)))

	for matrix_row in range(img_2_rows):
		for matrix_col in range(img_2_cols):
			neighbours = calculate_neighbours((matrix_row, matrix_col), 
											  (img_2_rows, img_2_cols),
											  shape,
											  radius)

			neighbourhood_average = 0
			for neighbour in neighbours:
				neighbourhood_average += pix_img_2[neighbour[0], neighbour[1]]
				
			training_set.append((np.array(neighbourhood_average / len(neighbours)), 
								np.array(0)))

	return training_set

def cross_validation(n_folders, size_folder):
	'''
	Function to calculate the accuracy of the algorithm
	'''
	pass

def main():
	image_1 = 'img_1.png'
	image_2 = 'img_2.png'

	training_set = generate_trainig_set(image_1, image_2)


if __name__ == '__main__':
	main()