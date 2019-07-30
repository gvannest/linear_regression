#!/usr/bin/env python3

import sys
import argparse
import json
import numpy as np

import py_files.algo_linear as algo_linear
from py_files.inputs import Data
from py_files.interval import Interval
from py_files.class_plot import GraphLive


def ft_errors(message):
	print(message)
	sys.exit(0)


def ft_argparser():

	def range_iter(value_string):
		value = int(value_string)
		if value not in range(0, 5001):
			raise argparse.ArgumentTypeError(f"{value} is out of range, choose in [0-5000]")
		return value

	def range_conf(value_string):
		value = float(value_string)
		if value not in range(0, 101):
			raise argparse.ArgumentTypeError(f"{value} is out of range, choose in [0-100]")
		return value

	def range_gen(value_string):
		value = int(value_string)
		if value not in range(0, 1001):
			raise argparse.ArgumentTypeError(f"{value} is out of range, choose in [0-1001]")
		return value

	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the training examples which will feed the linear_regression algorithm")
	parser.add_argument("-m", "--method", type=str, default="BGD", choices=["BGD", "MBGD", "SGD"], help="Type of gradient descent algorithm: Batch GD, Mini-Batch GD or Stochastic GD")
	parser.add_argument("-i", "--iterations", type=range_iter, default=150, choices=range(0, 5001), metavar="[0, 5000]", help="Fix number of iterations. Capped at 5000.")
	parser.add_argument("-a", "--alpha", type=float, default=1, help="Fix size of Gradient Descent step.")
	parser.add_argument("-p", "--plot", action="store_true", help="Draw a plot of cost function as GD advances. you can combine with flag -auto for autoscaling.")
	parser.add_argument("-auto", "--autoscale", action="store_true", help="When -p is True, autoscale the y_axis of the cost function")
	parser.add_argument("-c", "--confidence", type=range_conf, required=False, choices=range(0, 101), metavar="[0, 100]", help="Undertake an error analysis and use it to build a confidence interval with the given level of confidence")
	parser.add_argument("-g", "--generator", type=range_gen, required=False, choices=range(1, 1001), metavar="[1, 1000]", help="Add randomly generated data points around regression line.")
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


def ft_dump_data(model, prediction_interval):
	dic_data = {}
	dic_data['theta'] = list(model.theta)
	if prediction_interval:
		dic_data['p_interval'] = prediction_interval.p_interval
		dic_data['conf_level'] = prediction_interval.confidence
	with open('trained_data.json', 'w') as file:
		json.dump(dic_data, file)
	return None


def main(args):

	array_lines = np.genfromtxt(args.data_file, delimiter=',', skip_header=1)

	algo_linear.Algo.flag_plot = True if args.plot else False
	algo_linear.Algo.gd_algo = args.method
	GraphLive.auto_scale = args.autoscale

	data = Data(array_lines)
	data.normal_equation()
	if args.generator:
		data = ft_create_data(data)

	model = algo_linear.Algo(X=data.features, y=data.price, true_theta=data.true_theta)
	model.fit_linear(alpha=args.alpha, iter=args.iterations)

	prediction_interval = None
	if args.confidence:
		prediction_interval = Interval(model.theta, data.features, data.price, args.confidence/100)
		if args.plot:
			prediction_interval.graphs_residual()

	ft_dump_data(model, prediction_interval)

	return None


if __name__ == "__main__":
	np.set_printoptions(suppress=True)
	args = ft_argparser()
	main(args)