import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pyautogui


class Interval:

	def __init__(self, theta, features, price, confidence):
		self.theta = theta
		self.features = features
		self.true_price = price
		self.confidence = confidence
		self.price_estimates = np.dot(self.features, self.theta)
		self.residuals = self.true_price - self.price_estimates
		self.p_interval = self.calculate_pinterval()

	def get_mu_sigma_residuals(self):
		df = (self.true_price.size - 2)
		mu = np.mean(self.residuals) / df
		sum_errors = np.sum(np.square(self.residuals))
		sigma = np.sqrt(sum_errors / df)
		return mu, sigma

	def graphs_residual(self):

		def initialization(ax, title, x_label, y_label):
			ax.set_ylabel(y_label)
			ax.set_xlabel(x_label)
			ax.set_title(title)

		def graph_scatter(ax):
			initialization(ax, "Residuals given mileage", "Mileage", "Residuals")
			limit = max(np.max(self.residuals) * 1.05, abs(np.min(self.residuals) * 1.05))
			ax.set_ylim(-limit, limit)
			ax.scatter(self.features[:,-1], self.residuals, c='blue', label='residuals')
			ax.plot(np.array([0, 1.2 * np.max(self.features[:,-1], axis=0)]), np.zeros(2), '-', color="black", alpha=0.8, label='0 line')
			ax.legend(loc='upper right')

		def graph_histogram(ax):
			initialization(ax, "Density histogram of residuals", "Residual", "Probability density")
			mu, sigma = self.get_mu_sigma_residuals()

			n, bins, patches = ax.hist(self.residuals, bins='auto', density=1, edgecolor='white', linewidth=1.2)
			y_fit = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((bins - mu) / sigma) ** 2))
			ax.plot(bins, y_fit, '--', color='blue', label='Normal best fit')
			plt.axvline(x=mu, color='black', linewidth=1.2, label='Residual sample mean')
			ax.legend(loc='upper right')


		def plot_with_interval(ax):
			initialization(ax, "Regression line with prediction interval", "Mileage", "Price")
			ax.scatter(self.features[:,-1], self.true_price, c='blue')

			x_reg = np.array([0, 1.1 * np.max(self.features[:,-1], axis=0)])
			y_reg = self.theta[0] + self.theta[1] * x_reg
			p1 = ax.plot(x_reg, y_reg, '-', alpha=0.8)

			upper_bound = y_reg + self.p_interval
			lower_bound = y_reg - self.p_interval
			ax.fill_between(x_reg, upper_bound, lower_bound, color='orange', alpha='0.2')
			p2 = ax.fill(np.NaN, np.NaN, 'tab:orange', alpha=0.5)
			string_legend = f"Regression line with {self.confidence * 100:.0f} % prediction interval"
			ax.legend([(p2[0], p1[0]), ], [string_legend],loc='upper right')



		fig = plt.figure(figsize=(pyautogui.size()[0] / 96, pyautogui.size()[1] / 96))
		graph_scatter(fig.add_subplot(221))
		graph_histogram(fig.add_subplot(222))
		plot_with_interval(fig.add_subplot(212))
		plt.tight_layout(4, h_pad=3)
		plt.show()


	def calculate_pinterval(self):
		_, stdev = self.get_mu_sigma_residuals()
		Z_score = st.norm.ppf((self.confidence + 1) / 2)
		return Z_score * stdev





