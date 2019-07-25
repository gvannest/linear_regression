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

	def live_line_evolution(self, y_limit=None, x_limit=None, pause_time=0.002):
		if not self.line:
			plt.ion()
			self.initialization(y_limit, x_limit)
			self.line, = self.ax.plot(self.x_vec, self.y_vec, '-', alpha=0.8)
			plt.tight_layout(4, h_pad=3)
			plt.show()
		self.line.set_ydata(self.y_vec)
		plt.pause(pause_time)
		return None

	def live_regression(self, y_limit=None, x_limit=None, theta=None, true_theta=None, pause_time=0.002):
		x_reg = np.array([0, 1.2 * np.max(self.x_vec, axis=0)])
		y_reg = theta[0] + theta[1] * x_reg
		if not self.line:
			plt.ion()
			self.initialization(y_limit, x_limit)
			self.line, = self.ax.plot(x_reg, y_reg, '-', alpha=0.8)
			if true_theta is not None:
				y_true = true_theta[0] + true_theta[1] * x_reg
				line_true, = self.ax.plot(x_reg, y_true, '-', color="green", alpha=0.3)
			self.ax.scatter(self.x_vec, self.y_vec, c='blue')
			plt.tight_layout(4, h_pad=3)
			plt.show()
		self.line.set_ydata(y_reg)
		plt.pause(pause_time)
		return None

	def draw_contour(self, f, theta=None, pause_time=0.002):
		if not self.line:
			plt.ion()
			y_min = np.min(self.y_vec)
			y_max = np.max(self.y_vec)
			x_min = np.min(self.x_vec)
			x_max = np.max(self.x_vec)
			self.initialization(y_limit=(y_min, y_max), x_limit=(x_min, x_max))
			T0, T1 = np.meshgrid(self.x_vec, self.y_vec)
			Z = np.array([f(np.array([t0,t1])) for t0, t1 in zip(np.ravel(T0), np.ravel(T1))])
			Z = Z.reshape(T0.shape)
			self.ax.contour(T0, T1, Z, 20, cmap='RdGy')
			self.line, = self.ax.plot(theta[0], theta[1], '-', alpha=0.8)
			plt.show()
		x_data = self.line.get_xdata()
		y_data = self.line.get_ydata()
		self.line.set_xdata(np.r_[x_data, theta[0]])
		self.line.set_ydata(np.r_[y_data, theta[1]])

		plt.pause(pause_time)
		return None



	@classmethod
	def close(cls):
		plt.ioff()
		plt.show()

