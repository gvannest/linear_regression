import numpy as np
import matplotlib.pyplot as plt

class Interval:

	def __init__(self, theta, features, price):
		self.theta = theta
		self.features = features
		self.true_price = price
		self.price_estimates = np.dot(self.theta, self.features)
		self.residuals = np.square(self.true_price - self.price_estimates)
		self.prediction_interval = self.calculate_pinterval()


	def graph_residual(self):

		def graph_scatter():

		def graph_histogram()


	def calculate_pinterval(self):

	def plot_with_interval(self):



