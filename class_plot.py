import numpy as np

import matplotlib.pyplot as plt

class GraphLive:

	graph_count = 0
	plt.style.use('ggplot')
	fig = plt.figure(figsize=(20, 13))

	def __init__(self, x_vec=None, y_vec=None, ax=None, title=None, x_label=None, y_label=None):
		self.x_vec = x_vec
		self.y_vec = y_vec
		self.line = []
		self.ax = ax
		self.title = title
		self.x_label = x_label
		self.y_label = y_label
		GraphLive.graph_count += 1

	def initialization(self, y_limit=None, x_limit=None):
		self.ax.set_ylim(y_limit)
		self.ax.set_xlim(x_limit)
		self.ax.set_ylabel(self.y_label)
		self.ax.set_xlabel(self.x_label)
		self.ax.set_title(self.title)

	def live_line_evolution(self, y_limit=None, x_limit=None, pause_time=0.005):
		if not self.line:
			plt.ion()
			self.initialization(y_limit, x_limit)
			self.line, = self.ax.plot(self.x_vec, self.y_vec, '-', alpha=0.8)
			plt.tight_layout(4, h_pad=3)
			plt.show()
		self.line.set_ydata(self.y_vec)
		plt.pause(pause_time)
		return None

	def live_regression(self, y_limit=None, x_limit=None, theta=None, pause_time=0.005):
		x_reg = np.array([0, 1.2 * np.max(self.x_vec, axis=0)])
		y_reg = theta[0] + theta[1] * x_reg
		if not self.line:
			plt.ion()
			self.initialization(y_limit, x_limit)
			self.line, = self.ax.plot(x_reg, y_reg, '-', alpha=0.8)
			self.ax.scatter(self.x_vec, self.y_vec, c='blue')
			plt.tight_layout(4, h_pad=3)
			plt.show()
		self.line.set_ydata(y_reg)
		plt.pause(pause_time)
		return None

	def live_gd(self, f, pause_time=0.005):
		max_y = np.max(self.y_vec, axis=0)
		min_y = np.min(self.y_vec, axis=0)
		y_range = abs(max_y - min_y)
		theta0_range = np.arange(-1.2 * max_y, 1.2 * max_y)
		theta1_range = np.arange(-y_range, y_range)

		x1, x0 = np.meshgrid(theta1_range, theta0_range)
		plt.contour(x1, x0, f(self.x_vec,(x1,x0)), 20, cmap='RdGy')

	@classmethod
	def close(cls):
		plt.ioff()
		plt.show()

