import numpy as np

import matplotlib.pyplot as plt

from class_plot import GraphLive

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
		print(mean_)
		return X_norm, mean_, range_

	def predict(self, X, theta):
		return np.dot(X, theta)

	def cost(self, X, theta):
		return np.sum(np.square(self.predict(X, theta) - self.y)) / (2 * X.shape[0])

	def fit_linear(self, alpha=1, iter=200):
		m = self.X.shape[0]

		if Algo.flag_plot:
			g1 = GraphLive(x_vec=np.arange(0, iter), y_vec=np.full(iter, np.nan),
						   ax=GraphLive.fig.add_subplot(311), title="Real time cost evolution",
						   x_label="Iterations", y_label="J_history")
			g2 = GraphLive(x_vec=self.X[:,1], y_vec=self.y, ax=GraphLive.fig.add_subplot(312),
						   title="Regression line", x_label="Mileage", y_label="Price")
			g3 = GraphLive(x_vec=self.X_norm, y_vec=self.y, ax=GraphLive.fig.add_subplot(313),
						   title="Gradient descent", x_label="theta1", y_label="theta0")

		for i in range(iter):
			diff = np.dot(self.predict(self.X_norm, self.theta_norm) - self.y, self.X_norm)
			self.theta_norm = self.theta_norm - (alpha / m) * diff
			self.J_history.append(self.cost(self.X_norm, self.theta_norm))

			if Algo.flag_plot:
				self.theta[0] = self.theta_norm[0] - self.theta_norm[1] * self.mean_ / self.range_
				self.theta[1] = self.theta_norm[1] / self.range_
				g1.y_vec[i] = self.J_history[-1]
				g1.live_line_evolution(y_limit=(0, self.cost(self.X_norm, self.theta_norm)), x_limit=(0, iter))
				g2.live_regression(y_limit=(0, 1.2 * np.max(self.y, axis=0)),
								   x_limit=(0, 1.2 * np.max(self.X[:,1], axis=0)),
								   theta=self.theta)
				g3.live_gd(self.cost)

		self.theta[0] = self.theta_norm[0] - self.theta_norm[1] * self.mean_ / self.range_
		self.theta[1] = self.theta_norm[1] / self.range_
		if Algo.flag_plot:
			GraphLive.close()

		return None


