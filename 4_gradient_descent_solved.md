The following line will import the matplotlib library and enable plots to appear in the body of the notebook.


```python
%pylab inline --no-import-all
```

    Populating the interactive namespace from numpy and matplotlib


# 4 Function minimization

At this week's lecture we discussed how learning a set of weights (aka parameters) 
can be treated as the task of minimizing an error function. 

Scipy privides a number of ways of finding minima of arbitrary functions. We'll test one of them.


```python
import numpy
from scipy.optimize import fmin_bfgs, fmin
```

The function `fmin_bfgs` uses the function optimization method called 
[BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).
We need to give it the following arguments:
- f: the function to minimize
- x0: the initial guess of the argument with respect to which we're minimizing
- fprime: the first derivative of f. If we omit it, `fmin_bfgs` will use a numerical approximatiom

For example, consider the following polynomial function: $f(x) = x^4 - 10x^3 + x^2 + x - 4$




```python
def f(x):
    return x**4 - 10*x**3 + x**2 + x -4

def fprime(x):
    """First derivative of f(x) = x^4 - 10x^3 + x^2 + x - 4"""
    return 4*x**3 - 30*x**2 + 2*x + 1
```


```python
x = numpy.linspace(-2,10,1000)
pylab.plot(x,f(x))
```




    [<matplotlib.lines.Line2D at 0x7f1ff15a1828>]




![png](output_6_1.png)


Let's find the minimum, using 0 as a starting point:


```python
print(fmin_bfgs(f, x0=0, fprime=fprime))
```

    Optimization terminated successfully.
             Current function value: -4.093250
             Iterations: 5
             Function evaluations: 8
             Gradient evaluations: 8
    [-0.15101745]


Now let's omit the exact derivative of f and let `fmin_bfgs` use a numerical approximation of the derivative. The results should be close.


```python
print(fmin_bfgs(f, x0=0))
```

    Optimization terminated successfully.
             Current function value: -4.093250
             Iterations: 5
             Function evaluations: 24
             Gradient evaluations: 8
    [-0.15101746]


We can suppress the messages using disp=False:


```python
print(fmin_bfgs(f, x0=0, disp=False))
```

    [-0.15101746]


### Exercise 4.1
Looks like we got stuck in a local miniumum. Print the values of `x` where `f(x)` has a minimum findable from the following starting points: -2, 0, 2, 6, 10


```python
# .............................
for x0 in [-2, 0, 2, 6, 10]:
    x_min = fmin_bfgs(f, x0=x0, disp=False)
    print("Starting at {} found minimum at {}".format(x0,x_min))
```

    Starting at -2 found minimum at [-0.15101775]
    Starting at 0 found minimum at [-0.15101746]
    Starting at 2 found minimum at [ 7.42815758]
    Starting at 6 found minimum at [ 7.42815774]
    Starting at 10 found minimum at [ 7.4281578]



We can also use `fmin_bfgs` to minimize functions which take vectors rather than single numbers. For example, let's find the $x_1$ and $x_2$ which minimize the function $g(\mathbf{x}) = x_1^2 + x_2^2$ .


```python
def g(x):
    return (x**2).sum()
x0 = numpy.array([-1, -1])
x1 = numpy.array([1,1])
print("Starting at {} found minimum at {}".format(x0, fmin_bfgs(g, x0=x0, disp=False)))
print("Starting at {} found minimum at {}".format(x0, fmin_bfgs(g, x0=x1, disp=False)))
```

    Starting at [-1 -1] found minimum at [ -7.13245452e-09  -7.13245452e-09]
    Starting at [-1 -1] found minimum at [ -1.07505143e-08  -1.07505143e-08]


### Exercise 4.2

In this exercise we will regress the fourth feature of the iris dataset agains the first three, using error function minimization. First prepare the data. We will use the function `train_test_split` from `sklearn` to split the data into training and validation portions. The named argument `random_state=` sets the random seed and makes sure we will get the same split every time.


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_val, y_train, y_val = train_test_split(iris.data[:,:3], 
                                                  iris.data[:,3], 
                                                  test_size=1/3, 
                                                  random_state=666)
```

Now define the error function:


```python
def error(wb):
    '''Returns the error as a function of intercept and coefficients'''
    # We'll get the intercept as the fist element of wb, and the coefficients as the rest
    b = wb[0]
    w = wb[1:]
    # complete the function by returning the sum squared error 
    # .........................................
    return numpy.sum((X_train.dot(w)+b - y_train)**2)
```

Let's define a starting point for the intercept and coefficients


```python
# First item is the intercept b, the rest the coefficients w
wb0 = numpy.array([0.0, 0.0, 0.0, 0.0])
#................................
```

Now we are ready to find the values which minimize the error.


```python
wb_min = fmin_bfgs(error, x0=wb0)
print(wb_min)
```

    Optimization terminated successfully.
             Current function value: 2.869665
             Iterations: 5
             Function evaluations: 54
             Gradient evaluations: 9
    [-0.26721215 -0.2234636   0.24483245  0.53565652]


### Exercise 5.3

Let's check how well these parameters do on validation data, in terms of mean absolute error and r-squared. 


```python
from sklearn.metrics import r2_score, mean_absolute_error
# ..........................................

y_val_pred = X_val.dot(wb_min[1:])+wb_min[0]
print(mean_absolute_error(y_val, y_val_pred))
print(r2_score(y_val, y_val_pred))
```

    0.167035251859
    0.915157770009


### Exercise 4.4
Let us compare these results with the classic implementation of linear regression


```python
from sklearn.linear_model import LinearRegression
#.........................
model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_, model.coef_)
# 
print(mean_absolute_error(y_val, model.predict(X_val)))
print(r2_score(y_val, model.predict(X_val)))
```

    -0.267216604604 [-0.22346222  0.24483198  0.53565595]
    0.167035265248
    0.91515778547


## Stochastic Gradient Descent

Scikit learn provides two classes `SGDRegressor` and `SGDClassifier` which use stochastic gradient descent to carry out linear regression and classification respectively.

These models are especially useful in these situations:
- with very large datasets
- with streaming data (they support online learning)
- with datasets with sparse features

SGD is sensitive to learning rate and the scale of the features. It's strongly recommended to z-score the features, and to tune the learning rate.
Z-scoring refers to subtracting the mean and dividing by standard deviation from each feature. The `sklearn` class `StandardScaler` can be used for this purpose.


```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
```

We'll use the dataset of 50,000 songs. The prediction task is to guess the year the song was made based on 90 timbre features extracted from the audio. The year is in the first column.


```python
songs = numpy.load("/srv/songs50k.npy")
X_train, X_val, y_train, y_val = train_test_split(songs[:,1:], songs[:,0], test_size=1/3, random_state=666)
```

### Exercise 4.5

Train and evaluate the SGD classifier on the songs dataset. Take the following steps:
- z-score the training and validation features
- find good settings for learning rate type and learning rate initial value.
  - r-squared on validation data 
  - mean absolute error on validation data  


```python
# .........................................
scaler = StandardScaler()
X_train_z = scaler.fit_transform(X_train)
X_val_z  =scaler.transform(X_val)

lr = [ 10 ** x for x in range(-6,1)]

settings = []

for learning_rate in ['constant', 'optimal', 'invscaling']:
    for eta0 in lr:
        model = SGDRegressor(learning_rate=learning_rate, eta0=eta0, random_state=666)
        model.fit(X_train_z, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val_z))
        r2 =  r2_score(y_val, model.predict(X_val_z))
        settings.append((learning_rate, eta0, mae, r2))
        print(settings[-1])
best_mae = min(settings, key=lambda x: x[-2])
best_r2 =  max(settings, key=lambda x: x[-1])
print("Best settings according to MAE {}".format(best_mae))
print("Best settings according to R2 {}".format(best_r2))
```

    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('constant', 1e-06, 1691.6561284907273, -25739.655818085714)
    ('constant', 1e-05, 377.22562656494529, -1279.8415865574711)
    ('constant', 0.0001, 6.7478229230927438, 0.20522463589343209)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('constant', 0.001, 7.04352735572078, 0.1627636632877062)
    ('constant', 0.01, 88718907378.043488, -1.4427179523362913e+20)
    ('constant', 0.1, 3300084771630.748, -1.965778115631481e+23)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('constant', 1, 43545826379328.523, -3.7604428696391498e+25)
    ('optimal', 1e-06, 1669278438928.6868, -4.9318872872905672e+22)
    ('optimal', 1e-05, 1669278438928.6868, -4.9318872872905672e+22)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('optimal', 0.0001, 1669278438928.6868, -4.9318872872905672e+22)
    ('optimal', 0.001, 1669278438928.6868, -4.9318872872905672e+22)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('optimal', 0.01, 1669278438928.6868, -4.9318872872905672e+22)
    ('optimal', 0.1, 1669278438928.6868, -4.9318872872905672e+22)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('optimal', 1, 1669278438928.6868, -4.9318872872905672e+22)
    ('invscaling', 1e-06, 1976.6348192409769, -35142.50209656205)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('invscaling', 1e-05, 1790.3583459722679, -28830.97721901234)
    ('invscaling', 0.0001, 665.1822644785866, -3979.7739913242212)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('invscaling', 0.001, 6.8986085860230952, 0.17762923487895466)
    ('invscaling', 0.01, 6.8635494438831248, 0.19402007278906186)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('invscaling', 0.1, 9.7836377467991706, -0.66675789690938458)
    ('invscaling', 1, 1894875278231.2595, -6.9332563344365288e+22)
    Best settings according to MAE ('constant', 0.0001, 6.7478229230927438, 0.20522463589343209)
    Best settings according to R2 ('constant', 0.0001, 6.7478229230927438, 0.20522463589343209)


### Exercise 4.6

By default SGDRegressor tries to minimize the standard linear regression error function, that is sum of squared error. However this can be changed, via the `loss=` parameter. When `loss='squared_loss'`, sum of squared errors will be used. Other error functions available include [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) (`loss='huber'`). Compared to squared loss, huber focuses less on outliers.
![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Huber_loss.svg/600px-Huber_loss.svg.png)

Repeat the steps from the previous exercise, but include the tuning of the loss function.


```python
#................................................
scaler = StandardScaler()
X_train_z = scaler.fit_transform(X_train)
X_val_z  =scaler.transform(X_val)

lr = [ 10 ** x for x in range(-6,1)]

settings = []

for learning_rate in ['constant', 'optimal', 'invscaling']:
  for loss in ['squared_loss', 'huber']:
    for eta0 in lr:
        model = SGDRegressor(learning_rate=learning_rate, eta0=eta0, loss=loss,random_state=666)
        model.fit(X_train_z, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val_z))
        r2 =  r2_score(y_val, model.predict(X_val_z))
        settings.append((learning_rate, eta0, loss, mae, r2))
        print(settings[-1])
best_mae = min(settings, key=lambda x: x[-2])
best_r2 =  max(settings, key=lambda x: x[-1])
print("Best settings according to MAE {}".format(best_mae))
print("Best settings according to R2 {}".format(best_r2))
```

    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('constant', 1e-06, 'squared_loss', 1691.6561284907273, -25739.655818085714)
    ('constant', 1e-05, 'squared_loss', 377.22562656494529, -1279.8415865574711)
    ('constant', 0.0001, 'squared_loss', 6.7478229230927438, 0.20522463589343209)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('constant', 0.001, 'squared_loss', 7.04352735572078, 0.1627636632877062)
    ('constant', 0.01, 'squared_loss', 88718907378.043488, -1.4427179523362913e+20)
    ('constant', 0.1, 'squared_loss', 3300084771630.748, -1.965778115631481e+23)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('constant', 1, 'squared_loss', 43545826379328.523, -3.7604428696391498e+25)
    ('constant', 1e-06, 'huber', 1998.4742436817385, -35923.358210986873)
    ('constant', 1e-05, 'huber', 1998.3242451802123, -35917.965849589207)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('constant', 0.0001, 'huber', 1996.8242600653864, -35864.064492987789)
    ('constant', 0.001, 'huber', 1981.8243990365597, -35327.276670331383)
    ('constant', 0.01, 'huber', 1831.8248712503346, -30181.978245270922)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('constant', 0.1, 'huber', 331.77671963398245, -990.17942181017224)
    ('constant', 1, 'huber', 8.8194368200753814, -0.28066662934988074)
    ('optimal', 1e-06, 'squared_loss', 1669278438928.6868, -4.9318872872905672e+22)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('optimal', 1e-05, 'squared_loss', 1669278438928.6868, -4.9318872872905672e+22)
    ('optimal', 0.0001, 'squared_loss', 1669278438928.6868, -4.9318872872905672e+22)
    ('optimal', 0.001, 'squared_loss', 1669278438928.6868, -4.9318872872905672e+22)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('optimal', 0.01, 'squared_loss', 1669278438928.6868, -4.9318872872905672e+22)
    ('optimal', 0.1, 'squared_loss', 1669278438928.6868, -4.9318872872905672e+22)
    ('optimal', 1, 'squared_loss', 1669278438928.6868, -4.9318872872905672e+22)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('optimal', 1e-06, 'huber', 6.4966092908993458, 0.15129863635527019)
    ('optimal', 1e-05, 'huber', 6.4966092908993458, 0.15129863635527019)
    ('optimal', 0.0001, 'huber', 6.4966092908993458, 0.15129863635527019)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('optimal', 0.001, 'huber', 6.4966092908993458, 0.15129863635527019)
    ('optimal', 0.01, 'huber', 6.4966092908993458, 0.15129863635527019)
    ('optimal', 0.1, 'huber', 6.4966092908993458, 0.15129863635527019)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('optimal', 1, 'huber', 6.4966092908993458, 0.15129863635527019)
    ('invscaling', 1e-06, 'squared_loss', 1976.6348192409769, -35142.50209656205)
    ('invscaling', 1e-05, 'squared_loss', 1790.3583459722679, -28830.97721901234)
    ('invscaling', 0.0001, 'squared_loss', 665.1822644785866, -3979.7739913242212)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('invscaling', 0.001, 'squared_loss', 6.8986085860230952, 0.17762923487895466)
    ('invscaling', 0.01, 'squared_loss', 6.8635494438831248, 0.19402007278906186)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('invscaling', 0.1, 'squared_loss', 9.7836377467991706, -0.66675789690938458)
    ('invscaling', 1, 'squared_loss', 1894875278231.2595, -6.9332563344365288e+22)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('invscaling', 1e-06, 'huber', 1998.4898106315125, -35923.917857523171)
    ('invscaling', 1e-05, 'huber', 1998.4799146789337, -35923.562091108273)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('invscaling', 0.0001, 'huber', 1998.3809551509053, -35920.004523783573)
    ('invscaling', 0.001, 'huber', 1997.3913596464247, -35884.438532967586)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)


    ('invscaling', 0.01, 'huber', 1987.4953823072424, -35529.746872855743)
    ('invscaling', 0.1, 'huber', 1888.53350000253, -32079.660075372416)
    ('invscaling', 1, 'huber', 898.78868758109172, -7266.1691238908488)
    Best settings according to MAE ('optimal', 1e-06, 'huber', 6.4966092908993458, 0.15129863635527019)
    Best settings according to R2 ('constant', 0.0001, 'squared_loss', 6.7478229230927438, 0.20522463589343209)


    /home/gchrupala/.local/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)



```python

```
