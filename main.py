import os

from PIL import Image
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from img_segmentation import generate_trainig_set, get_neighbours_values
from mlp import MultiLayerPerceptron as MLP

class ImageSegmentation:
	def __init__(self, img_path):
		self._figure = plt.figure()
		self._setup_axes()
		self._setup_gui()

		self._object_img = None
		self._background_img = None

		self._area = []
		self._cropping = None

		self._mlp = MLP((4, 15, 5, 1))

		# Load target image in grayscale
		self._img = Image.open(img_path).convert('L')

		# Plot image
		self._img_ax.imshow(self._img, cmap='gray')

		# Set event listeners
		self._figure.canvas.mpl_connect('button_press_event', self._onclick)

	def _setup_axes(self):
		grid = gridspec.GridSpec(2, 5)

		self._img_ax = self._create_axes('Img. Original', grid[:, :2])
		self._object_ax = self._create_axes('Img. Objeto', grid[0:1, 2:3])
		self._background_ax = self._create_axes('Img. Fondo', grid[1:2, 2:3])
		self._result_ax = self._create_axes('Img. Segmentada', grid[:, 3:])

	def _setup_gui(self):
		self._object_btn = Button(self._figure.add_axes([0.1, 0.05, 0.2, 0.06]), 'Selec. Objeto')
		self._background_btn = Button(self._figure.add_axes([0.3, 0.05, 0.2, 0.06]), 'Selec. Fondo')
		self._training_btn = Button(self._figure.add_axes([0.5, 0.05, 0.2, 0.06]), 'Entrenar')
		self._test_btn = Button(self._figure.add_axes([0.7, 0.05, 0.2, 0.06]), 'Probar')

		self._object_btn.on_clicked(self._on_select_object)
		self._background_btn.on_clicked(self._on_select_background)
		self._training_btn.on_clicked(self._on_train)
		self._test_btn.on_clicked(self._on_test)

	def _create_axes(self, title, spec):
		axes = self._figure.add_subplot(spec)
		axes.set_title(title)
		axes.set_axis_off()

		return axes

	def _on_select_object(self, event):
		print 'Seleccionar objeto'

		self._cropping = 'object'

	def _on_select_background(self, event):
		print 'Seleccionar fondo'

		self._cropping = 'background'

	def _on_train(self, event):
		if self._object_img and self._background_img:
			images = [self._object_img, self._background_img]
			training_set = generate_trainig_set(images)

			self._mlp.train(
				training_set,
				learning_rate=0.05,
				max_epochs=200,
				min_error=0.003)

			self._plot_result()
		else:
			print 'Debes seleccionar el objeto y el fondo'

	def _on_test(self, event):
		results_folder_name = 'Results'
		test_folder = raw_input('Carpeta de imagenes a probar: ')
		test_images = []
		
		for file in os.listdir(test_folder):
			filename, ext = file.split('.')
			if ext == 'jpg' or ext == 'png' or ext == 'jpeg':
				test_images.append(file)

		if not os.path.exists(results_folder_name):
			os.makedirs(results_folder_name)	
		
		#Process all the images to test
		for image in test_images:
			new_test_image = Image.open(test_folder + '/' + image).convert('L')
			pixels = new_test_image.load()
			ncols, nrows = new_test_image.size

			#Process test image
			for col in range(ncols):
				for row in range(nrows):
					inputs = get_neighbours_values(new_test_image, (row, col))
					output, = self._mlp.test(inputs)

					if output == 0:
						pixels[col, row] = 255
					else:
						pixels[col, row] = 0
			
			#save segmented image
			filename, ext = image.split('.')
			new_image_name = filename + '__result__.' + ext
			new_test_image.save(results_folder_name + '/' + new_image_name)

		print 'Pruebas terminadas'
			
	def _plot_result(self):
		result_img = self._img.copy()
		pixels = result_img.load()

		ncols, nrows = result_img.size

		for col in range(ncols):
			for row in range(nrows):
				inputs = get_neighbours_values(self._img, (row, col))
				output, = self._mlp.test(inputs)

				if output == 0:
					pixels[col, row] = 1
				else:
					pixels[col, row] = 0
		self._result_ax.imshow(result_img, cmap='gray')

	def _onclick(self, event):
		if (self._cropping
			and event.inaxes
			and event.inaxes == self._img_ax):

			# Get position
			x = int(event.xdata)
			y = int(event.ydata)

			self._area.extend([x, y])

			if len(self._area) == 4:
				try:
					self._crop()
					self._exit_cropping_mode()
				except:
					print 'Error al recortar imagen!'

				self._area = []

	def _crop(self):
		cropped = self._img.crop(self._area)

		if self._cropping == 'object':
			self._object_img = cropped
			self._object_ax.imshow(cropped, cmap='gray')
		else:
			self._background_img = cropped
			self._background_ax.imshow(cropped, cmap='gray')

		self._figure.canvas.draw()

	def _exit_cropping_mode(self):
		self._cropping = None

if __name__ == '__main__':
	segmentation = ImageSegmentation('1.jpg')
	plt.show()
