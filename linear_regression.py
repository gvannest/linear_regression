import sys
import argparse

import numpy as np

from inputs import Data

def ft_errors(message):
	print(message)
	sys.exit(0)

def ft_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument("data_file", type=str, help="csv file containing the training examples which will feed the linear_regression algorithm")
	# parser.add_argument("-i", "--interactive", action="store_true", help="Interactive facts mode, where the user can change facts or add new facts")
	# parser.add_argument("-u", "--undetermined", action="store_true", help="Undetermined mode, where the user can clarify undetermined facts")
	# parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode. Outputs the rules leading to a particular conclusion")
	args = parser.parse_args()
	return args


def main(args):

	with open(args.data_file) as file:
		lines = file.readlines()

	data = Data(lines)
	print(data.km)
	print(data.price)
	print(data.features)
	print(data.theta)

	return None



if __name__ == "__main__":
	args = ft_argparser()
	main(args)