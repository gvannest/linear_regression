import sys
import argparse

import numpy as np

from inputs import Data
import algo_linear


def ft_errors(message):
	print(message)
	sys.exit(0)


def ft_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the training examples which will feed the linear_regression algorithm")
	parser.add_argument("-m", "--method", type=str, default="BGD", choices=["BGD", "MBGD", "SGD"], help="Type of gradient descent algorithm: Batch GD, Mini-Batch GD or Stochastic GD")
	parser.add_argument("-i", "--iterations", type=int, default=150, help="fix number of iterations")
	parser.add_argument("-a", "--alpha", type=float, default=1, help="fix size of Gradient Descent step")
	parser.add_argument("-p", "--plot", action="store_true", help="Draw a plot of cost function as GD advances")
	args = parser.parse_args()
	return args


def main(args):

	array_lines = np.genfromtxt(args.data_file, delimiter=',', skip_header=1)

	algo_linear.Algo.flag_plot = True if args.plot else False
	algo_linear.Algo.gd_algo = args.method

	data = Data(array_lines)
	data.normal_equation()
	print(data.true_theta)

	model = algo_linear.Algo(X=data.features, y=data.price, true_theta=data.true_theta)
	model.fit_linear(alpha=args.alpha, iter=args.iterations)
	print(model.theta)

	return None


if __name__ == "__main__":
	np.set_printoptions(suppress=True)
	args = ft_argparser()
	main(args)