import numpy as np


class Algo:

	def __init__(self, X=None, y=None):
		self.X = X
		self.y = y
		self.theta = np.zeros(self.X.shape[1], dtype=int)
		self.X_norm, self.mean_, self.range_ = self.feature_normalization()
		self.J_history = [self.cost()]

	def feature_normalization(self):
		mean_ = np.sum(self.X[:, 1:]) / self.X.shape[0]
		range_ = np.max(self.X[:, 1:], axis=0) - np.min(self.X[:, 1:], axis=0)
		X_norm = np.c_[self.X[:,0], (self.X[:, 1:] - mean_) / range_]
		return X_norm, mean_, range_

	def predict(self):
		return np.dot(self.X_norm, self.theta)

	def cost(self):
		return np.sum(np.square(self.predict() - self.y)) / (2 * self.X.shape[0])

	def fit_linear(self, alpha=0.01, iter=500):
		m = self.X.shape[0]
		for _ in range(iter):
			diff = np.dot(self.predict() - self.y, self.X_norm)
			self.theta = self.theta - (alpha / m) * diff
			self.J_history.append(self.cost())

		return None
