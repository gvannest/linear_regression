import numpy as np

class Data:

	def __init__(self, array_lines):
		self.km = array_lines[:,0]
		self.price = array_lines[:,1]
		self.features = np.c_[np.ones(len(self.km)), self.km]
		self.true_theta = None

	def normal_equation(self):
		self.true_theta = np.linalg.inv((np.transpose(self.features).dot(self.features)))\
			.dot(np.transpose(self.features))\
			.dot(self.price)

		return None

	def update_data(self, new_km, new_prices):
		self.km = np.append(self.km, new_km)
		self.price = np.append(self.price, new_prices)
		self.features = np.c_[np.ones(len(self.km)), self.km]
		self.normal_equation()
		return None




