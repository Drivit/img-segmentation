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
			else:
				neighbours_matrix.append((-1, -1))
			#Down
			if (matrix_row) + i < max_row:
				neighbours_matrix.append((matrix_col, matrix_row+i))
			else:
				neighbours_matrix.append((-1, -1))
			#Left
			if (matrix_col) - i >= 0:
				neighbours_matrix.append((matrix_col-i, matrix_row))
			else:
				neighbours_matrix.append((-1, -1))
			#Right
			if (matrix_col) + i < max_col:
				neighbours_matrix.append((matrix_col+i, matrix_row))
			else:
				neighbours_matrix.append((-1, -1))

	elif shape == 'square':
		pass
		'''
		for row in range(radius):
			for col in range(radius):
				if (col < matrix_col):
					neighbours_matrix.append((matrix_col+col, matrix_row))
		'''
				
	return neighbours_matrix

def get_neighbours_values(img, matrix_position, radius=1):
	pix_img = img.load()
	max_col, max_row = img.size

	neighbours = calculate_neighbours(matrix_position, 
									  (max_row, max_col),
									  'cross',
									  radius)

	neighbourhood_average = 0
	total_real_neighbours = len(neighbours)

	for neighbour in neighbours:
		if neighbour[0] != -1:
			neighbourhood_average += pix_img[neighbour[0], neighbour[1]]
		else:
			total_real_neighbours -= 1

	neighbourhood_average /= total_real_neighbours
	total_neighbours = len(neighbours)
	
	# Calculate gray scales array
	array_gray_values = np.zeros(total_neighbours)
	for i in range(total_neighbours):
		if neighbours[i][0] != -1:
			array_gray_values[i] = pix_img[neighbours[i][0], neighbours[i][1]]
		else:
			array_gray_values[i] = neighbourhood_average

	return np.array(array_gray_values)
	

def generate_trainig_set(images, shape='cross', radius=1):
	'''
	Function to create a training set from two images

	Parameters
	----------
	img1: isntance of an image from PIL.Image, this image is for class 1
	img2: isntance of an image from PIL.Image, this image is for class 0
	shape: shape that will be used in the neighbourhood, 'cross' as default
	radius: neighbourhood's block radius, 1 as default
	'''

	training_set = []
	class_id = 0

	for image in images:
		# Extract pixels maps
		pix_img = image.load()
	
		# Get image size
		img_cols, img_rows  =  image.size

		# Create training set from image
		for matrix_row in range(img_rows):
			for matrix_col in range(img_cols):
				neighbours = calculate_neighbours((matrix_row, matrix_col), 
												  (img_rows, img_cols), 
												  shape,
												  radius)

				# Calculate neighbourhood average to fill empty neighbours positions
				neighbourhood_average = 0
				total_real_neighbours = len(neighbours)
				for neighbour in neighbours:
					if neighbour[0] != -1:
						neighbourhood_average += pix_img[neighbour[0], neighbour[1]]
					else:
						total_real_neighbours -= 1

				neighbourhood_average /= total_real_neighbours
				total_neighbours = len(neighbours)
				
				# Add gray scales array to training set
				array_gray_values = np.zeros(total_neighbours)
				for i in range(total_neighbours):
					if neighbours[i][0] != -1:
						array_gray_values[i] = pix_img[neighbours[i][0], neighbours[i][1]]
					else:
						array_gray_values[i] = neighbourhood_average

				training_set.append( (np.array(array_gray_values), np.array(class_id)) )

		class_id += 1

	return training_set

if __name__ == '__main__':
	img_1 = Image.open('image_1.png').convert('L')
	img_2 = Image.open('image_2.jpg').convert('L')

	training_set = generate_trainig_set([img_1, img_2])