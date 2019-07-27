import sys
import argparse

import numpy as np

import algo_linear
from inputs import Data
from interval import Interval


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
	parser.add_argument("-c", "--confidence", type=float, required=False, choices=np.arange(0, 101, 1) / 100, metavar="[0, 1]", help="Undertake an error analysis and use it to build a confidence interval with the given level of confidence")
	parser.add_argument("-g", "--generator", type=int, required=False, choices=range(1, 1001), metavar="[1, 1000]", help="Add randomly generated data points around regression line.")
	args = parser.parse_args()
	return args


def ft_create_data(data):
	if args.generator <= 1000:
		new_km = np.random.randint(np.min(data.km), 1.05 * np.max(data.km), size=args.generator)
		noise = np.random.normal(0, 1, args.generator) * 700
		new_prices = data.true_theta[0] + data.true_theta[1] * new_km + noise
		data.update_data(new_km, new_prices)
	else:
		ft_errors("Error : data points generator is capped to 1000.")

	return data


def main(args):

	array_lines = np.genfromtxt(args.data_file, delimiter=',', skip_header=1)

	algo_linear.Algo.flag_plot = True if args.plot else False
	algo_linear.Algo.gd_algo = args.method

	data = Data(array_lines)
	data.normal_equation()
	if args.generator:
		data = ft_create_data(data)

	model = algo_linear.Algo(X=data.features, y=data.price, true_theta=data.true_theta)
	model.fit_linear(alpha=args.alpha, iter=args.iterations)
	if args.confidence:
		prediction_interval = Interval(model.theta, data.features, data.price, args.confidence)
		prediction_interval.graphs_residual()


	return None


if __name__ == "__main__":
	np.set_printoptions(suppress=True)
	args = ft_argparser()
	main(args)