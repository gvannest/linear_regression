import sys
import json

def print_output(prediction, confidence=None, conf_level=None):
	print(f"Mean estimation of the price of your car : {prediction:.0f} euros")
	if confidence:
		print(f"Prediction interval at {conf_level * 100:.0f} % : [{prediction - confidence:.0f}, {prediction + confidence:.0f}]")
	return None

def	main(mileage):

	theta = [0,0]

	try:
		with open('trained_data.json') as file:
			data = json.load(file)
	except FileNotFoundError as e:
		print("Warning : it seems you have not trained the model. Thetas are set to zero.")
		return(print_output(0))
	else:
		theta = data['theta']
		prediction = theta[0] + theta[1] * mileage
		conf = None
		conf_level = None
		if "p_interval" in data.keys():
			conf = data["p_interval"]
			conf_level = data["conf_level"]
		return(print_output(prediction, conf, conf_level))

if __name__ == "__main__":
	try:
		mileage = int(input("Please provide mileage of your car :\n"))
	except ValueError as e:
		print(e)
		sys.exit(1)
	else:
		main(mileage)