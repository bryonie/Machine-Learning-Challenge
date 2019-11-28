# Logistic regression

In this session we will develop a simple implementation of Logistic Regression trained with SDG. The goal is to develop the understanding of gradient descent, the logistic regression model and the practical use of numpy.

First we'll load some toy data to use with our functions.  We'll make this into a binary problem by keeping only two species.


```python
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()

# skip rows with the label 2
data = iris.data[iris.target != 2]
target = iris.target[iris.target != 2]
X_train, X_val, y_train, y_val = train_test_split(data, target, 
                                                  test_size=1/3, random_state=123)


# Z-score the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print(X_train.shape)
```

    (66, 4)


## Model definition


We'll first define the interface of our model:

- `predict` - compute predicted classes on new examples given a trained model
- `predict_proba` - - compute class probabilities on new examples given a trained model
- `fit` - train a model using features and labels from the training set

as well some auxiliary functions.


### Exercise 6.1

Define function `inverse_logit`. The mathematical formulation is:
$$
\mathrm{logit}^{-1}(z) = \frac{1}{1+\exp(-z)}
$$



```python
def inverse_logit(z):
    #..................................
    return 1/(1+numpy.exp(-z))
```


```python
print(inverse_logit(0.5))
print(inverse_logit(-10.0))
print(inverse_logit(0.0))
print(inverse_logit(40.0))
print(inverse_logit(40.0) == inverse_logit(100.0))
```

    0.622459331202
    4.53978687024e-05
    0.5
    1.0
    True


(Due to limited precision of floating point numbers, past a certain absolute value of the input, our function becomes a constant 1 or 0.)

### Exercise 6.2 

Define function `predict_proba`, with two arguments:

- dictionary of model parameters `{'w':w,'b':b}`, where `w` is an numpy array of coefficients and `b` a scalar intercept
- numpy array (matrix) of new the features of new examples `X`

The function should return an array of probabilities of the positive class.


```python
def predict_proba(wb, X):
    #...............................
    return inverse_logit(X.dot(wb['w']) + wb['b'])
```


```python
# Initial model parameters
w = numpy.zeros((X_train.shape[1],))
b = 0
wb = {'w':w,'b':b}
# Use this initial model for prediction
p_pred = predict_proba(wb, X_val)
print(X_val.shape)
print(p_pred.shape)
print(p_pred)
```

    (34, 4)
    (34,)
    [ 0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5
      0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5
      0.5  0.5  0.5  0.5]


### Exercise 6.3
Define function `predict` which takes the same input as `predict_proba` but returns the class labels (0 or 1) instead of probabilities.


```python
def predict(wb, X):
    #.............................
    return (predict_proba(wb, X) >= 0.5).astype('int')
```


```python
y_pred = predict(wb, X_val)
print(X_val.shape)
print(y_pred.shape)
print(y_pred)
```

    (34, 4)
    (34,)
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]


Our model interface is complete.

## Training
We will now implement the interface of the SGD training algorithm:

- `fit` which takes initial model parameters and trains it for one pass over the given training data

We will start with an auxiliary function `update` which does a single step of SGD.


### Exercise 6.5

Define function `update` which is given a single training example, and first uses the `predict_proba` function to get the predicted probability of the positive class, and then updates the weights and the bias of
the model depending on the difference between this probability and the actual target. 

The function is given these arguments:

- `wb` - the model weights and bias (dictionary of model parameters `{'w':w,'b':b}`, where `w` is an numpy array of coefficients and `b` a scalar intercept)
- `x`  - the feature vector of the training example
- `y`  - the class label of the training example
- `eta`- learning rate

The update should change the given parameters by implementing the following operations:
$$
\mathbf{w}_{new} = \mathbf{w}_{old} + \eta(y-p_{pred})\mathbf{x}
$$

and

$$
b_{new} = b_{old} + \eta (y-p_{pred})
$$

Finally, the function should return the value of the loss for the current examples, that is:
$$
-y \log_2(p_{pred}) - (1-y)\log_2(1-p_{pred})
$$



```python
def update(wb, x, y, eta):
    #.............................
    p_pred = predict_proba(wb, x)
    wb['w'] += eta*(y-p_pred)*x
    wb['b'] += eta*(y-p_pred)
    
    return -y*numpy.log2(p_pred)-(1-y)*numpy.log2(1-p_pred)
```


```python
from pprint import pprint
wb = {'w':numpy.zeros((X_train.shape[1],)), 'b':0}
eta = 0.1
# Show P(y=1) before and after update

# Process example 1
i = 0
print("Actual class: {}".format(y_train[i]))
print("P(y=1): {:.3}".format(predict_proba(wb, X_train[i])))
loss = update(wb, X_train[i], y_train[i], eta)
print("Loss: {:.3}".format(loss))
pprint(wb)
print("P(y=1): {:.3}".format(predict_proba(wb, X_train[i])))


print()
# Process example 5
i = 5
print("Actual class: {}".format(y_train[i]))
print("P(y=1): {:.3}".format(predict_proba(wb, X_train[i])))
loss = update(wb, X_train[i], y_train[i], eta)
print("Loss: {:.3}".format(loss))
pprint(wb)
print("P(y=1): {:.3}".format(predict_proba(wb, X_train[i])))



```

    Actual class: 1
    P(y=1): 0.5
    Loss: 1.0
    {'b': 0.050000000000000003,
     'w': array([-0.00510916, -0.00941896,  0.05861284,  0.06567975])}
    P(y=1): 0.552
    
    Actual class: 0
    P(y=1): 0.48
    Loss: 0.943
    {'b': 0.0020289335835076,
     'w': array([ 0.04832087, -0.00038221,  0.10706961,  0.12371601])}
    P(y=1): 0.423


### Exercise 6.5 

Define function `fit`, which will use the `update` function on each training example in turn, for a single iteration of SGD. The function takes the following arguments:

- `wb` - the current model weights and bias
- `X` - the matrix of training example features
- `y` - the vector of training example classes
- `eta=0.1` - the learning rate, with default 0.1

The function returns the sum of the losses on all the examples, as given by `update`.



```python
def fit(wb, X, y, eta=0.01):
    #..................................
    assert X.shape[0] == y.shape[0]
    # Explicit for loop
    loss = []
    for i in range(X.shape[0]):
        L = update(wb, X[i,:], y[i], eta)
        loss.append(L)
    return sum(loss)
```


```python
def fit(wb, X, y, eta=0.01):
    #..................................
    assert X.shape[0] == y.shape[0]
    # List comprehension
    return sum(update(wb, X[i,:], y[i], eta) for i in range(X.shape[0]))
```


```python
wb = {'w':numpy.zeros((4,)), 'b':0}
eta = 0.01
J = 10

# Let's run 10 epochs of SGD
print("epoch loss")
for j in range(J):
    loss = fit(wb, X_train, y_train, eta=0.1)
    print("{} {:.3}".format(j, loss))
```

    epoch loss
    0 17.7
    1 4.63
    2 2.82
    3 2.05
    4 1.62
    5 1.34
    6 1.15
    7 1.0
    8 0.89
    9 0.801


### Exercise 6.6

Train your model for one pass (epoch) and check classification accuracy on validation data.


```python
from sklearn.metrics import accuracy_score

model = {'w':numpy.zeros((4,)), 'b':0}
fit(model, X_train, y_train, eta=0.1)
acc = accuracy_score(predict(model, X_val), y_val)
print("Accuracy: {:.3}".format(acc))
```

    Accuracy: 1.0


## SGD classifier 

SGD classifier is suitable to use on large datasets, as well as on sparse data such as character or word ngram counts.

We'll use the scikit-learn implementation of Logistic Regression with SGD to learn to classify posts on various discussion groups into topics.  There are twenty groups:


```python
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
for group in data.target_names:
    print(group)
```

    alt.atheism
    comp.graphics
    comp.os.ms-windows.misc
    comp.sys.ibm.pc.hardware
    comp.sys.mac.hardware
    comp.windows.x
    misc.forsale
    rec.autos
    rec.motorcycles
    rec.sport.baseball
    rec.sport.hockey
    sci.crypt
    sci.electronics
    sci.med
    sci.space
    soc.religion.christian
    talk.politics.guns
    talk.politics.mideast
    talk.politics.misc
    talk.religion.misc


The data is in the form of raw text, so we'll need to extract some features from it.


```python
print(data.data[0])
```

    I was wondering if anyone out there could enlighten me on this car I saw
    the other day. It was a 2-door sports car, looked to be from the late 60s/
    early 70s. It was called a Bricklin. The doors were really small. In addition,
    the front bumper was separate from the rest of the body. This is 
    all I know. If anyone can tellme a model name, engine specs, years
    of production, where this car is made, history, or whatever info you
    have on this funky looking car, please e-mail.


We will split the data into train and validation, and then extract word counts from the texts.


```python
text_train, text_val, y_train, y_val = train_test_split(data.data, data.target, test_size=1/2, random_state=123)
```


```python
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(analyzer='word', ngram_range=(1,1), lowercase=True)
X_train = vec.fit_transform(text_train)
X_val = vec.transform(text_val)
```

We can now try the SGDClassifier on this data.


```python
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
```


```python
model = SGDClassifier(loss='log', random_state=666)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("{:.3}".format(accuracy_score(y_val, y_pred)))
```

    0.603


### Exercise 6.7

Experiment with different features and model hyperparameters, and find a well performing setting.


```python

```
