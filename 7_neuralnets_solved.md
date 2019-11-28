# Introduction to neural networks


In this session we will start with a simple toy implementation of a neural network and apply it to the XOR problem. In the second part we will learn how to use the [Keras toolkit](https://keras.io/) to define, train and use a practical neural network model.

## XOR

Let's start with the [XOR problem](https://en.wikipedia.org/wiki/XOR_gate). 


```python
import numpy
%pylab inline --no-import-all
```

    Populating the interactive namespace from numpy and matplotlib


### Exercise 7.1
Define the function `xor`, which which takes a Nx2 array, where each row is an input to the logical XOR. It outputs an array of size N with the corresponding outputs.


```python
def xor(X):
    #.........
    return (X.sum(axis=1) == 1).astype(int)
```


```python
X = numpy.array([[0, 0],      # FALSE
                 [0, 1],      # TRUE
                 [1, 0],      # TRUE
                 [1, 1]])     # FALSE
y = xor(X)
print(y)
```

    [0 1 1 0]



```python
pylab.scatter(X[:,0], X[:,1], c=y, s=200)
```




    <matplotlib.collections.PathCollection at 0x7fa1563cf860>




![png](output_7_1.png)


## Neural network
We can define a simple two layer neural network by hand which solves the XOR classification problem. The network has parameters $\mathbf{W}$ and $\mathbf{U}$, and computes the following:

$$Y = \sigma(U(\sigma(WX^T))$$

Where $\mathbf{X}$ is the input array, with shape Nx2, $\mathbf{W}$ is a 2x2 matrix, and $\mathbf{U}$ is a 1x2 matrix. The result is a 1xN matrix (i.e. a single row vector) of XOR values.

### Exercise 7.2

Define function `sigma` which returns one if the input is greater than or equal to 0.5, and zero otherwise.


```python
def sigma(X):
    #...............
    return (X >= 0.5).astype(float)
```


```python
z = numpy.random.uniform(0,1,(3,2))
print(z)
print(sigma(z))
```

    [[ 0.95125363  0.83903522]
     [ 0.75394668  0.28882588]
     [ 0.80585408  0.58343813]]
    [[ 1.  1.]
     [ 1.  0.]
     [ 1.  1.]]


### Exercise 7.3

Define function `nnet` which takes the weight matrices W and U, and the input X, and returns the result Y computed according to the formula above.


```python
def nnet(W,U,X):
    #..........................................
    Z = sigma(numpy.dot(W,numpy.transpose(X)))
    return sigma(numpy.dot(U,Z))
```

Define the weights:


```python
W = numpy.array([[1,-1],
                 [-1,1]])
U = numpy.array([1,1])
```

Check what it outputs


```python
y_pred = nnet(W, U, X)
print(y)
print(y_pred)

```

    [0 1 1 0]
    [ 0.  1.  1.  0.]


And plot the outputs as a function of inputs.


```python
# Create a grid of points for plotting
shape=(20,20)
grid = numpy.array([ [i,j] for i in numpy.linspace(0,1,shape[0]) 
                               for j in numpy.linspace(0,1,shape[1]) ])
# Apply the neural net to all the points
y_pred = nnet(W, U, grid)
pylab.pcolor(y_pred.reshape((20,20)))
pylab.colorbar()
pylab.xticks([])
pylab.yticks([])
```




    ([], <a list of 0 Text yticklabel objects>)




![png](output_18_1.png)


## Training XOR NN with Keras

We'll now learn how to build a simple neural network in Keras. 


```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam

model = Sequential()
# Add two hidden layers with 4 hidden units each, and the tanh activation.

model.add(Dense(4, input_dim=2, activation='tanh'))
model.add(Dense(4, activation='tanh'))

# The final layer is the output layer with an inverse logit activation function.
model.add(Dense(1, activation='sigmoid'))

# Use the Adam optimizer. Adam works similar to regular SGD, 
# but with some important improvements: https://arxiv.org/abs/1412.6980
optimizer = Adam(lr=0.02)
model.compile(optimizer=optimizer, loss='binary_crossentropy')
```

    Using TensorFlow backend.


We can now train the model, specifying number of epochs, size of the minibatch, and whether to print extra information.


```python
model.fit(X, y, epochs=100, batch_size=1, verbose=1)
```

    Epoch 1/100
    4/4 [==============================] - 0s 81ms/step - loss: 0.7507
    Epoch 2/100
    4/4 [==============================] - 0s 17ms/step - loss: 0.7023
    Epoch 3/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.6902
    Epoch 4/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.6845
    Epoch 5/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.6846
    Epoch 6/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.6737
    Epoch 7/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.6748
    Epoch 8/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.6676
    Epoch 9/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.6652
    Epoch 10/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.6616
    Epoch 11/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.6590
    Epoch 12/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.6523
    Epoch 13/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.6462
    Epoch 14/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.6370
    Epoch 15/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.6364
    Epoch 16/100
    4/4 [==============================] - 0s 3ms/step - loss: 0.6289
    Epoch 17/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.6214
    Epoch 18/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.6102
    Epoch 19/100
    4/4 [==============================] - 0s 3ms/step - loss: 0.5960
    Epoch 20/100
    4/4 [==============================] - 0s 3ms/step - loss: 0.5864
    Epoch 21/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.5769
    Epoch 22/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.5675
    Epoch 23/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.5581
    Epoch 24/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.5439
    Epoch 25/100
    4/4 [==============================] - 0s 3ms/step - loss: 0.5380
    Epoch 26/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.5208
    Epoch 27/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.5130
    Epoch 28/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.5007
    Epoch 29/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.4919
    Epoch 30/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.4854
    Epoch 31/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.4684
    Epoch 32/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.4539
    Epoch 33/100
    4/4 [==============================] - 0s 18ms/step - loss: 0.4468
    Epoch 34/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.4337
    Epoch 35/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.4201
    Epoch 36/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.4007
    Epoch 37/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.3892
    Epoch 38/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.3689
    Epoch 39/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.3518
    Epoch 40/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.3311
    Epoch 41/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.3275
    Epoch 42/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.3081
    Epoch 43/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.2858
    Epoch 44/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.2622
    Epoch 45/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.2432
    Epoch 46/100
    4/4 [==============================] - 0s 22ms/step - loss: 0.2240
    Epoch 47/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.2053
    Epoch 48/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.1883
    Epoch 49/100
    4/4 [==============================] - 0s 22ms/step - loss: 0.1698
    Epoch 50/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.1570
    Epoch 51/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.1418
    Epoch 52/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.1284
    Epoch 53/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.1171
    Epoch 54/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.1050
    Epoch 55/100
    4/4 [==============================] - 0s 22ms/step - loss: 0.0949
    Epoch 56/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0854
    Epoch 57/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0792
    Epoch 58/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0729
    Epoch 59/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0668
    Epoch 60/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0598
    Epoch 61/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0560
    Epoch 62/100
    4/4 [==============================] - 0s 19ms/step - loss: 0.0504
    Epoch 63/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0468
    Epoch 64/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.0440
    Epoch 65/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.0403
    Epoch 66/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0377
    Epoch 67/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0358
    Epoch 68/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.0330
    Epoch 69/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0309
    Epoch 70/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0291
    Epoch 71/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0276
    Epoch 72/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.0260
    Epoch 73/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0246
    Epoch 74/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0235
    Epoch 75/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0223
    Epoch 76/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0213
    Epoch 77/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0206
    Epoch 78/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.0194
    Epoch 79/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0185
    Epoch 80/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0177
    Epoch 81/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0170
    Epoch 82/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0163
    Epoch 83/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0157
    Epoch 84/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0152
    Epoch 85/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0145
    Epoch 86/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0140
    Epoch 87/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0136
    Epoch 88/100
    4/4 [==============================] - 0s 20ms/step - loss: 0.0131
    Epoch 89/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0127
    Epoch 90/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0123
    Epoch 91/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0119
    Epoch 92/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0115
    Epoch 93/100
    4/4 [==============================] - 0s 1ms/step - loss: 0.0112
    Epoch 94/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0108
    Epoch 95/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0106
    Epoch 96/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0102
    Epoch 97/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0099
    Epoch 98/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0097
    Epoch 99/100
    4/4 [==============================] - 0s 21ms/step - loss: 0.0094
    Epoch 100/100
    4/4 [==============================] - 0s 2ms/step - loss: 0.0091





    <keras.callbacks.History at 0x7fa145b7ad30>




```python
print("   x1          x2          F(x1, x2)")
print(np.hstack([X, model.predict(X)]))
```

       x1          x2          F(x1, x2)
    [[ 0.          0.          0.00521935]
     [ 0.          1.          0.99282151]
     [ 1.          0.          0.99291104]
     [ 1.          1.          0.01610738]]



```python
# Apply the neural net to all the points
y_pred = model.predict(grid)
pylab.pcolor(y_pred.reshape((20,20)))
pylab.colorbar()
pylab.xticks([])
pylab.yticks([])
```




    ([], <a list of 0 Text yticklabel objects>)




![png](output_24_1.png)


## Regression with NN on iris

We will now define and train a neural network model for regression on the iris data.

### Load data


```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data = load_iris()
# Inputs
X = numpy.array(data.data[:,0:3], dtype='float32')
# Output
y = numpy.array(data.data[:,3], dtype='float32')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/3, random_state=999)
print(X_train.shape)
print(y_train.shape)
```

    (100, 3)
    (100,)


### Exercise 7.4


Define a multilayer perceptron with the following specifications:
- Hidden layer 1: size 16, activation: tanh
- Hidden layer 2: size 16, activation: tanh
- Output layer: size 1, activation: linear

Compile it using the following specifications:
- optimizer: Adam
- loss: mean squared error

Train the network, and try to find a good value of learning rate by monitoring the loss.

Compute mean absolute error and r-squared the validation data.


```python
#..................................

model = Sequential()
model.add(Dense(16, input_dim=3, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(1, activation='linear'))

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='mean_squared_error', optimizer=optimizer)

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=2)
```

    Epoch 1/20
     - 1s - loss: 0.4570
    Epoch 2/20
     - 0s - loss: 0.3616
    Epoch 3/20
     - 0s - loss: 0.2963
    Epoch 4/20
     - 0s - loss: 0.2395
    Epoch 5/20
     - 0s - loss: 0.1939
    Epoch 6/20
     - 0s - loss: 0.1559
    Epoch 7/20
     - 0s - loss: 0.1271
    Epoch 8/20
     - 0s - loss: 0.1047
    Epoch 9/20
     - 0s - loss: 0.0890
    Epoch 10/20
     - 0s - loss: 0.0762
    Epoch 11/20
     - 0s - loss: 0.0682
    Epoch 12/20
     - 0s - loss: 0.0627
    Epoch 13/20
     - 0s - loss: 0.0581
    Epoch 14/20
     - 0s - loss: 0.0565
    Epoch 15/20
     - 0s - loss: 0.0545
    Epoch 16/20
     - 0s - loss: 0.0536
    Epoch 17/20
     - 0s - loss: 0.0519
    Epoch 18/20
     - 0s - loss: 0.0507
    Epoch 19/20
     - 0s - loss: 0.0500
    Epoch 20/20
     - 0s - loss: 0.0496





    <keras.callbacks.History at 0x7fa118f71438>




```python
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_val)
print(mean_absolute_error(y_val, y_pred))
print(r2_score(y_val, y_pred))
```

    0.181786
    0.900227223893


## Classification

Let's now do classification. The target is a categorical vector. It will need to be transformed to an array of dummies. This transform is also called on-hot encoding.
This can be done manually, but sklearn.preprocessing has some utilities that make it simple:
- OneHotEncoder
- LabelBinarizer



```python
# Inputs
X = numpy.array(data.data, dtype='float32')
# Output
y = numpy.array(data.target, dtype='int32')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/3, random_state=999)

# One-hot Indicator array for classes
from sklearn.preprocessing import LabelBinarizer
onehot = LabelBinarizer()
Y_train = onehot.fit_transform(y_train)
Y_val   = onehot.transform(y_val)

print(Y_train[:10,:])
```

    [[0 0 1]
     [0 1 0]
     [1 0 0]
     [0 0 1]
     [1 0 0]
     [1 0 0]
     [1 0 0]
     [0 1 0]
     [0 0 1]
     [0 0 1]]


### Exercise 7.5

Define a multilayer perceptron with the following specifications:
- Hidden layer 1: size 16, activation: tanh
- Hidden layer 2: size 16, activation: tanh
- Output layer: size 3, activation: softmax

NB: softmax is a generalization of inverse logit to more than 2 classes. It converts class scores to class probabilities, while making sure than they sum up to 1:

```
def softmax(x):
    z = numpy.exp(x)
    return z/numpy.sum(z)
```

Compile it using the following specifications:
- optimizer: Adam
- loss: categorical_crossentropy

Train the network, and try to find a good value of learning rate by monitoring the loss.
Use the method `.predict_classes` to predict the targets on validation data.
Compute the classification accuracy on validation data.


```python
#.....................................
model = Sequential()
model.add(Dense(16, input_dim=4, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(3, activation='softmax')) # We need to have as many units as classes, 
                                                             # and softmax activation
optimizer = Adam(lr=0.001)
# For classification, the loss function should be categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=2)
```

    Epoch 1/10
     - 1s - loss: 0.9960
    Epoch 2/10
     - 0s - loss: 0.6772
    Epoch 3/10
     - 0s - loss: 0.4705
    Epoch 4/10
     - 0s - loss: 0.3259
    Epoch 5/10
     - 0s - loss: 0.2696
    Epoch 6/10
     - 0s - loss: 0.2153
    Epoch 7/10
     - 0s - loss: 0.1671
    Epoch 8/10
     - 0s - loss: 0.1619
    Epoch 9/10
     - 0s - loss: 0.1375
    Epoch 10/10
     - 0s - loss: 0.1181





    <keras.callbacks.History at 0x7fa118c4a6a0>




```python
#.....................................
from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(X_val, verbose=False)
print(accuracy_score(y_val, y_pred))
```

    0.96


### Exercise 7.6


Train a neural network classifier on the handwritten digits dataset. 
This dataset comes with scikit learn and can be accessed as follows:


```python
from sklearn.datasets import load_digits
digits = load_digits()
images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:10]):
    pylab.subplot(2, 5, index + 1)
    pylab.axis('off')
    pylab.imshow(image,cmap=plt.cm.gray_r)
    pylab.title('%i' % label)
```


![png](output_36_0.png)


The targets are in `digits.target` and the pixel values flattened into an array are in `digits.data`.

Train a classifier on the first 1000 of the images, and evaluate on the rest. 
Before testing the neural network model, check the classification error rate of a logistic regression classifier as a baseline.


Remember to convert the targets to the one-hot representation for training the neural network.

Some things to try when training a neural network model for this dataset:

- start with two or three hidden layers
- use between 32 to 128 units in each layer
- try different learning rates in the Adam optimizer (lr=0.001, lr=0.0001) and monitor the loss function
- train for at least 100 epochs
- try the `relu` activation function instead of `tanh`




```python
# .....................
X_train = digits.data[:1000,:]
y_train = digits.target[:1000]
X_val = digits.data[1000:,:]
y_val = digits.target[1000:]

onehot = LabelBinarizer()
Y_train = onehot.fit_transform(y_train)
Y_val   = onehot.transform(y_val)

from sklearn.linear_model import LogisticRegression
baseline = LogisticRegression()
baseline.fit(X_train, y_train)
print(1-accuracy_score(y_val, baseline.predict(X_val)))
```

    0.0740276035132



```python
#.....................................
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(Y_train.shape[1], activation='softmax')) # We need to have as many units as classes, 
                                                             # and softmax activation
optimizer = Adam(lr=0.001)
# For classification, the loss function should be categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=0)
```




    <keras.callbacks.History at 0x7fa118096cf8>




```python
y_pred = model.predict_classes(X_val, verbose=0)
print(1-accuracy_score(y_val, y_pred))
```

    0.0677540777917



```python

```
