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

```
./linear_regression.py [-h] [-m {BGD,MBGD,SGD}] [-i [0, 5000]] [-a ALPHA]
                       [-p] [-auto] [-c [0, 100]] [-g [1, 1000]]
                        data_file
```

**Positional arguments (= required):**

Argument         | Description              
:----------------|:-----------------------
data_file        | csv file containing the training examples which will feed the linear_regression algorithm|
  

**Optional arguments :**

|Short flag            | Long flag              | Description             |
:----------------------|:-----------------------| :-----------------------|
  -h                   | --help                 | Show help message
  -m {BGD, MBGD, SGD}  | --method             | Type of gradient descent algorithm. Choices are  : BGD (Batch GD), MBGD (Mini-Batch GD) or SGD (Stochastic GD)
  -i   [0-5000]        | --iterations           | Fix number of iterations. Capped at 5000.
  -a                   | --alpha                | Fix size of Gradient Descent step.
  -p                   | --plot                 | Draw a real time plot of cost function, gradient descent and linear fit as GD advances. If -c is on, plot an analysis of residuals and prediction interval after the training. You can combine with flag -auto for autoscaling.
  -auto                | --autoscale            | When -p is True, autoscale the y_axis of the cost function (allows for better visualization in case of divergence)
  -c [0-100]           | --confidence           | Undertake an error analysis and use it to build a confidence interval with the given level of confidence
  -g [1-1000]          | --generator            | Add randomly generated data points around regression line.
  
  





