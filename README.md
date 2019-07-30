# linear_regression

Linear Regression is a 42 project where you have to build a program to estimate the price of a car given its mileage.  
The idea is to code a gradient descent algorithm based on a dataset of price/mileage cars.  
The cost function to minimize is the MSE (for Mean Squared Errors).

There are two programs in the repo. One 

## Input

The main input is a csv file of the form :

```
km,price
240000,3650
139800,3800
150500,4400
185530,4450
176000,5250
114800,5350
166800,5800
89000,5990
144500,5999
84000,6200
82029,6390
63060,6390
74000,6600
97500,6800
67000,6800
76025,6900
48235,6900
93000,6990
60949,7490
65674,7555
54000,7990
68500,7990
22899,7990
61789,8290
```

## Usage

You need to have python 3 installed on your machine.

```./computor [-h] [-r] [-t] [-v] [-n] input```

**Positional arguments (= required):**

Argument         | Description              
:----------------|:-----------------------
input            | The input string representing the equation to solve|
  

**Optional arguments :**

Short flag       | Long flag              | Description
:----------------|:-----------------------| :---------------------------|
  -h             | --help                 | Show help message
  -r             | --human_readable       | Present the solution in a more "reader-friendly" format (`^0` and `1 *` are excluded)  
  -t             | --tree                 | Allows for visualizing the solving tree in the terminal window.
  -v             | --verbose              | Prints out each calculation involved in reducing and solving the equation
  -n             | --neg                  | Tries to solve for negative degrees (multiplying each element by the absolute value of the lowest degree)

