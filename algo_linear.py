import numpy as np

import matplotlib.pyplot as plt

from class_plot import GraphLive
import linear_regression

class Algo:

	flag_plot = False
	gd_algo = ''
	mv_avg = 10
	batch_size = 10

	def __init__(self, X=None, y=None, true_theta=None):
		self.X = X
		self.y = y
		self.theta_norm = np.zeros(self.X.shape[1], dtype=float)
		self.theta = np.zeros(self.X.shape[1], dtype=float)
		self.X_norm, self.mean_, self.range_ = self.feature_normalization()
		self.J_history = []
		self.true_theta = true_theta

	def feature_normalization(self):
		mean_ = np.sum(self.X[:, 1:]) / self.X.shape[0]
		range_ = np.max(self.X[:, 1:], axis=0) - np.min(self.X[:, 1:], axis=0)
		X_norm = np.c_[self.X[:,0], (self.X[:, 1:] - mean_) / range_]
		return X_norm, mean_, range_

	def predict(self, X, theta):
		return np.dot(X, theta)

	def cost_bgd(self, theta):
		return np.sum(np.square(self.predict(self.X_norm, theta) - self.y)) / (2 * self.X_norm.shape[0])

	def cost_sgd(self, theta, e=0):
		return np.square(self.predict(self.X_norm[e], theta) - self.y[e]) / 2

	def cost_mbgd(self, X, y, theta):
		return np.sum(np.square(self.predict(X, theta) - y)) / (2 * X.size)

	def dynamic_plots(self, iter, i, **kwargs):
		self.theta[0] = self.theta_norm[0] - self.theta_norm[1] * self.mean_ / self.range_
		self.theta[1] = self.theta_norm[1] / self.range_

		if Algo.gd_algo == "SGD":
			kwargs['g1'].y_vec[i] = np.sum(np.array([self.J_history[-a] for a in range(1, Algo.mv_avg + 1)])) / Algo.mv_avg
		else:
			kwargs['g1'].y_vec[i] = self.J_history[-1]
		kwargs['g1'].live_line_evolution(y_limit=(0, self.J_history[0]), x_limit=(0, iter))
		kwargs['g2'].live_regression(y_limit=(0, 1.2 * np.max(self.y, axis=0)),
									 x_limit=(0, 1.2 * np.max(self.X[:, 1], axis=0)),
									 theta=self.theta, true_theta=self.true_theta)
		kwargs['g3'].draw_contour(self.cost_bgd, theta=self.theta_norm)

		return None


	def batch_gradient(self, alpha, iter, m, **kwargs):
		for i in range(iter):
			diff = np.dot(self.predict(self.X_norm, self.theta_norm) - self.y, self.X_norm)
			self.theta_norm = self.theta_norm - (alpha / m) * diff
			self.J_history.append(self.cost_bgd(self.theta_norm))

			if Algo.flag_plot:
				self.dynamic_plots(iter, i, **kwargs)

		return None


	def stochastic_gradient(self, alpha, iter, m, **kwargs):

		def learning_rate_decay(epoch):
			return 1/ (3 + epoch)

		index_array = np.arange(m)
		run = 0
		for i in range(iter):
			np.random.shuffle(index_array)
			for e in index_array:
				run += 1
				diff = np.dot(self.predict(self.X_norm[e], self.theta_norm) - self.y[e], self.X_norm[e])
				self.J_history.append(self.cost_sgd(self.theta_norm, e))
				self.theta_norm = self.theta_norm - alpha * diff
			alpha = learning_rate_decay(i)

			if Algo.flag_plot :
				self.dynamic_plots(iter, i, **kwargs)
		return None


	def mb_gradient(self, alpha, iter, m, **kwargs):

		def ft_shuffle_data():
			concat_array = np.c_[self.X_norm, self.y]
			np.random.shuffle(concat_array)
			self.X_norm = concat_array[:, :-1]
			self.y = concat_array[:, -1]

		def ft_get_batch(X, y, b=Algo.batch_size):
			if b > m:
				linear_regressioin.ft_errors("Batch size cannot exceed number of training examples.")
			start = 0
			batch_i = b
			X_batch = []
			y_batch = []
			while batch_i <= m:
				X_batch.append(X[start:batch_i,:])
				y_batch.append(y[start:batch_i])
				batch_i += b
				start += b
			if start < m:
				X_batch.append(X[start:,:])
				y_batch.append(y[start:])
			for x,y in zip(X_batch, y_batch):
				yield x, y

		for i in range(iter):
			ft_shuffle_data()
			for X_batch, y_batch in ft_get_batch(self.X_norm, self.y):
				diff = np.dot(self.predict(X_batch, self.theta_norm) - y_batch, X_batch)
				self.J_history.append(self.cost_mbgd(X_batch, y_batch, self.theta_norm))
				self.theta_norm = self.theta_norm - (alpha / X_batch.size) * diff

			if Algo.flag_plot:
				self.dynamic_plots(iter, i, **kwargs)

		return None


	def fit_linear(self, alpha=1, iter=150):
		m = self.X.shape[0]

		if Algo.flag_plot:
			g1 = GraphLive(x_vec=np.arange(0, iter), y_vec=np.full(iter, np.nan),
						   ax=GraphLive.fig.add_subplot(221), title="Real time cost evolution",
						   x_label="Iterations", y_label="J_history")
			g2 = GraphLive(x_vec=self.X[:,1], y_vec=self.y, ax=GraphLive.fig.add_subplot(222),
						   title="Regression line", x_label="Mileage", y_label="Price")
			g3 = GraphLive(x_vec=np.arange(3000, 10000, 50),
						   y_vec=np.arange(-12000, 5000, 50), ax=GraphLive.fig.add_subplot(212),
						   title="Gradient descent", x_label="theta0", y_label="theta1")

		if Algo.gd_algo == "BGD":
			if Algo.flag_plot:
				self.batch_gradient(alpha, iter, m, g1=g1, g2=g2, g3=g3)
			else:
				self.batch_gradient(alpha, iter, m)
		elif Algo.gd_algo == "SGD":
			if Algo.flag_plot:
				self.stochastic_gradient(alpha, iter, m, g1=g1, g2=g2, g3=g3)
			else:
				self.stochastic_gradient(alpha, iter, m)
		elif Algo.gd_algo == "MBGD":
			if Algo.flag_plot:
				self.mb_gradient(alpha, iter, m, g1=g1, g2=g2, g3=g3)
			else:
				self.mb_gradient(alpha, iter, m)


		self.theta[0] = self.theta_norm[0] - self.theta_norm[1] * self.mean_ / self.range_
		self.theta[1] = self.theta_norm[1] / self.range_
		if Algo.flag_plot:
			GraphLive.close()

		return None


