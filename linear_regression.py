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
	parser.add_argument("-i", "--iterations", type=int, default=200, help="fix number of iterations")
	parser.add_argument("-a", "--alpha", type=float, default=1, help="fix size of Gradient Descent step")
	parser.add_argument("-p", "--plot", action="store_true", help="Draw a plot of cost function as GD advances")
	args = parser.parse_args()
	return args


def main(args):

	with open(args.data_file) as file:
		lines = file.readlines()

	Algo.flag_plot = True if args.plot else False
	data = Data(lines)
	model = Algo(X=data.features, y=data.price)
	model.fit_linear(alpha=args.alpha, iter=args.iterations)
	print(model.theta_norm)
	print(model.theta)

	return None


if __name__ == "__main__":
	np.set_printoptions(suppress=True)
	args = ft_argparser()
	main(args)