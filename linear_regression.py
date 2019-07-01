import sys
import argparse

import numpy as np

from inputs import Data
from algo_linear import Algo


def ft_errors(message):
	print(message)
	sys.exit(0)


def ft_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the training examples which will feed the linear_regression algorithm")
	# parser.add_argument("-i", "--interactive", action="store_true", help="Interactive facts mode, where the user can change facts or add new facts")
	args = parser.parse_args()
	return args


def main(args):

	with open(args.data_file) as file:
		lines = file.readlines()

	data = Data(lines)
	model = Algo(X=data.features, y=data.price)
	model.fit_linear(alpha=0.1, iter=1000)
	print(model.theta)
	#dump les theta en pkl. denormaliser les thetas avant

	return None


if __name__ == "__main__":
	np.set_printoptions(suppress=True)
	args = ft_argparser()
	main(args)