import numpy as np

class Data:

	def __init__(self, lines):
		self.km = np.asarray(self.parse_lines(lines)[0], dtype=int)
		self.price = np.asarray(self.parse_lines(lines)[1], dtype=int)
		self.features = np.c_[np.ones(len(self.km)), self.km]

	def parse_lines(self, lines):
		new_lines = [e.strip().split(',') for e in lines[1:]]
		list_km = [int(e[0]) for e in new_lines]
		list_price = [int(e[1]) for e in new_lines]
		return list_km, list_price




