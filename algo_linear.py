import numpy as np

import matplotlib.pyplot as plt

class Algo:

	flag_plot = False

	def __init__(self, X=None, y=None):
		self.X = X
		self.y = y
		self.theta_norm = np.zeros(self.X.shape[1], dtype=float)
		self.theta = np.zeros(self.X.shape[1], dtype=float)
		self.X_norm, self.mean_, self.range_ = self.feature_normalization()
		self.J_history = []

	def feature_normalization(self):
		mean_ = np.sum(self.X[:, 1:]) / self.X.shape[0]
		range_ = np.max(self.X[:, 1:], axis=0) - np.min(self.X[:, 1:], axis=0)
		X_norm = np.c_[self.X[:,0], (self.X[:, 1:] - mean_) / range_]
		return X_norm, mean_, range_

	def predict(self):
		return np.dot(self.X_norm, self.theta_norm)

	def cost(self):
		return np.sum(np.square(self.predict() - self.y)) / (2 * self.X.shape[0])

	def live_plot1(self, fig, x_vec1, y1, line1, iter, pause_time=0.05):
		if not line1:
			plt.ion()
			ax1 = fig.add_subplot(211)
			ax1.set_ylim([0, self.cost()])
			ax1.set_xlim([0, iter])
			line1, = ax1.plot(x_vec1, y1, '-', alpha=0.8)
			ax1.set_ylabel('J_history')
			ax1.set_xlabel('Iterations')
			ax1.set_title('Cost evolution')
			plt.tight_layout(4, h_pad=3)
			plt.show()
		line1.set_ydata(y1)
		plt.pause(pause_time)
		return line1

	def live_plot2(self, fig, x_vec2, y2, line2, pause_time=0.05):
		if not line2:
			plt.ion()
			ax2 = fig.add_subplot(212)
			line2, = ax2.plot(x_vec2, y2, '-', alpha=0.8)
			ax2.scatter(self.X[:,1], self.y, c='blue')
			ax2.set_ylabel('Price')
			ax2.set_xlabel('Mileage')
			ax2.set_title('Regression line')
			plt.tight_layout(4, h_pad=3)
			plt.show()
		line2.set_ydata(y2)
		plt.pause(pause_time)
		return line2

	def fit_linear(self, alpha=1, iter=200):
		m = self.X.shape[0]
		if Algo.flag_plot:
			plt.style.use('ggplot')
			fig = plt.figure(figsize=(13, 13))
			x_vec1 = np.arange(0, iter)
			x_vec2 = np.array([0, np.max(self.X[:,1], axis=0)])
			line1 = []
			line2 = []
			y1 = np.full(len(x_vec1), np.nan)
		for i in range(iter):
			diff = np.dot(self.predict() - self.y, self.X_norm)
			self.theta_norm = self.theta_norm - (alpha / m) * diff
			self.J_history.append(self.cost())
			if Algo.flag_plot:
				self.theta[0] = self.theta_norm[0] - self.theta_norm[1] * self.mean_ / self.range_
				self.theta[1] = self.theta_norm[1] / self.range_
				y2 = self.theta[0] + self.theta[1] * x_vec2
				y1[i] = self.J_history[-1]
				line1 = self.live_plot1(fig, x_vec1, y1, line1, iter)
				line2 = self.live_plot2(fig, x_vec2, y2, line2)
		self.theta[0] = self.theta_norm[0] - self.theta_norm[1] * self.mean_ / self.range_
		self.theta[1] = self.theta_norm[1] / self.range_
		if Algo.flag_plot:
			plt.ioff()
			plt.show()

		return None


