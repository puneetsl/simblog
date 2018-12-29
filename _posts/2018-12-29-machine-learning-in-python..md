
# Machine Learning Exercise 1 - Linear Regression

This notebook covers a Python-based solution for the first programming exercise of the machine learning class on Coursera.  Please refer to the [exercise text](https://github.com/jdwittenauer/ipython-notebooks/blob/master/exercises/ML/ex1.pdf) for detailed descriptions and equations.

In this exercise we'll implement simple linear regression using gradient descent and apply it to an example problem.  We'll also extend our implementation to handle multiple variables and apply it to a slightly more difficult example.

## Linear regression with one variable

In the first part of the exercise, we're tasked with implementing linear regression with one variable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities.

Let's start by importing some libraries and examining the data.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
import os
path = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 6.1101</td>
      <td> 17.5920</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 5.5277</td>
      <td>  9.1302</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 8.5186</td>
      <td> 13.6620</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 7.0032</td>
      <td> 11.8540</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 5.8598</td>
      <td>  6.8233</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td> 97.000000</td>
      <td> 97.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>  8.159800</td>
      <td>  5.839135</td>
    </tr>
    <tr>
      <th>std</th>
      <td>  3.869884</td>
      <td>  5.510262</td>
    </tr>
    <tr>
      <th>min</th>
      <td>  5.026900</td>
      <td> -2.680700</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>  5.707700</td>
      <td>  1.986900</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>  6.589400</td>
      <td>  4.562300</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>  8.578100</td>
      <td>  7.046700</td>
    </tr>
    <tr>
      <th>max</th>
      <td> 22.203000</td>
      <td> 24.147000</td>
    </tr>
  </tbody>
</table>
</div>



Let's plot it to get a better idea of what the data looks like.


```python
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0xd140198>




![png](assets/machine-learning-in-python_files/machine-learning-in-python_9_1.png)


Now let's implement linear regression using gradient descent to minimize the cost function.  The equations implemented in the following code samples are detailed in "ex1.pdf" in the "exercises" folder.

First we'll create a function to compute the cost of a given solution (characterized by the parameters theta).


```python
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
```

Let's add a column of ones to the training set so we can use a vectorized solution to computing the cost and gradients.


```python
data.insert(0, 'Ones', 1)
```

Now let's do some variable initialization.


```python
# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
```

Let's take a look to make sure X (training set) and y (target variable) look correct.


```python
X.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ones</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 1</td>
      <td> 6.1101</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 1</td>
      <td> 5.5277</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 1</td>
      <td> 8.5186</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 1</td>
      <td> 7.0032</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 1</td>
      <td> 5.8598</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 17.5920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>  9.1302</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 13.6620</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 11.8540</td>
    </tr>
    <tr>
      <th>4</th>
      <td>  6.8233</td>
    </tr>
  </tbody>
</table>
</div>



The cost function is expecting numpy matrices so we need to convert X and y before we can use them.  We also need to initialize theta.


```python
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
```

Here's what theta looks like.


```python
theta
```




    matrix([[0, 0]])



Let's take a quick look at the shape of our matrices.


```python
X.shape, theta.shape, y.shape
```




    ((97L, 2L), (1L, 2L), (97L, 1L))



Now let's compute the cost for our initial solution (0 values for theta).


```python
computeCost(X, y, theta)
```




    32.072733877455676



So far so good.  Now we need to define a function to perform gradient descent on the parameters theta using the update rules defined in the text.


```python
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
```

Initialize some additional variables - the learning rate alpha, and the number of iterations to perform.


```python
alpha = 0.01
iters = 1000
```

Now let's run the gradient descent algorithm to fit our parameters theta to the training set.


```python
g, cost = gradientDescent(X, y, theta, alpha, iters)
g
```




    matrix([[-3.24140214,  1.1272942 ]])



Finally we can compute the cost (error) of the trained model using our fitted parameters.


```python
computeCost(X, y, g)
```




    4.5159555030789118



Now let's plot the linear model along with the data to visually see how well it fits.


```python
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
```




    <matplotlib.text.Text at 0xd35a518>




![png](assets/machine-learning-in-python_files/machine-learning-in-python_37_1.png)


Looks pretty good!  Since the gradient decent function also outputs a vector with the cost at each training iteration, we can plot that as well.  Notice that the cost always decreases - this is an example of a convex optimization problem.


```python
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
```




    <matplotlib.text.Text at 0xd5bccc0>




![png](assets/machine-learning-in-python_files/machine-learning-in-python_39_1.png)


## Linear regression with multiple variables

Exercise 1 also included a housing price data set with 2 variables (size of the house in square feet and number of bedrooms) and a target (price of the house).  Let's use the techniques we already applied to analyze that data set as well.


```python
path = os.getcwd() + '\data\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Size</th>
      <th>Bedrooms</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 2104</td>
      <td> 3</td>
      <td> 399900</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 1600</td>
      <td> 3</td>
      <td> 329900</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 2400</td>
      <td> 3</td>
      <td> 369000</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 1416</td>
      <td> 2</td>
      <td> 232000</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 3000</td>
      <td> 4</td>
      <td> 539900</td>
    </tr>
  </tbody>
</table>
</div>



For this task we add another pre-processing step - normalizing the features.  This is very easy with pandas.


```python
data2 = (data2 - data2.mean()) / data2.std()
data2.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Size</th>
      <th>Bedrooms</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 0.130010</td>
      <td>-0.223675</td>
      <td> 0.475747</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.504190</td>
      <td>-0.223675</td>
      <td>-0.084074</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 0.502476</td>
      <td>-0.223675</td>
      <td> 0.228626</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.735723</td>
      <td>-1.537767</td>
      <td>-0.867025</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 1.257476</td>
      <td> 1.090417</td>
      <td> 1.595389</td>
    </tr>
  </tbody>
</table>
</div>



Now let's repeat our pre-processing steps from part 1 and run the linear regression procedure on the new data set.


```python
# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
computeCost(X2, y2, g2)
```




    0.13070336960771897



We can take a quick look at the training progess for this one as well.


```python
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
```




    <matplotlib.text.Text at 0xd7bb240>




![png](assets/machine-learning-in-python_files/machine-learning-in-python_48_1.png)


Instead of implementing these algorithms from scratch, we could also use scikit-learn's linear regression function.  Let's apply scikit-learn's linear regressio algorithm to the data from part 1 and see what it comes up with.


```python
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, normalize=False)



Here's what the scikit-learn model's predictions look like.


```python
x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
```




    <matplotlib.text.Text at 0xf258860>




![png](assets/machine-learning-in-python_files/machine-learning-in-python_52_1.png)


That's it!  Thanks for reading.  In Exercise 2 we'll take a look at logistic regression for classification problems.
