# Tensorflow tutorial: Quickstart for beginners


```python
import tensorflow as tf
```

Load & prepare MNIST


```python
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize data
X_train, X_test = X_train / 255.0, X_test / 255.0
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 14s 1us/step


Build a sequential neural net


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(10)
])
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 128)               100480    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________


For each example, the model returns logits, or log-odds scores for each class

**logits**: non-normalized predictions (usually input to softmax which normalizes the probabilities)<br> 

**log-odds**: logarithm of the odds of an event: $\text{log-odds} = \ln(\frac{p}{1-p})$. This is the inverse of sigmoid.


```python
predictions = model(X_train[:1]).numpy()
predictions
```




    array([[-0.12554444, -0.126252  , -0.5178235 ,  0.13532521, -0.50004005,
            -0.41123134, -0.35366565, -0.14402214, -0.13909012,  0.43870565]],
          dtype=float32)



Now use `tf.nn.softmax` to convert to probabilities


```python
tf.nn.softmax(predictions).numpy()
```




    array([[0.10066038, 0.10058919, 0.06799766, 0.13066305, 0.0692177 ,
            0.07564607, 0.08012846, 0.09881749, 0.09930606, 0.17697392]],
          dtype=float32)



**Note**: you can put tf.nn.softmax as the last layer. Discouraged b/c softmax output makes it impossible to provide an exact / stable loss calculation. 

`losses.SparseCategoricalCrossentropy` - take a vector of logits & returns a scalar loss


```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

This is the negative log probability of the true class: zero if model is 100% sure that this is the correct class. 

Since the model is untrained, the probabilities will be close to random, so loss should be log(1/10) ~ 2.3


```python
loss_fn(y_train[:1], predictions).numpy
```




    <bound method _EagerTensorBase.numpy of <tf.Tensor: shape=(), dtype=float32, numpy=2.5816898>>



Now compile and train the model


```python
model.compile(optimizer='adam', 
             loss=loss_fn, 
             metrics=['accuracy'])
```


```python
model.fit(X_train, y_train, epochs=5)
```

    Epoch 1/5
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.2986 - accuracy: 0.9131
    Epoch 2/5
    1875/1875 [==============================] - 2s 933us/step - loss: 0.1446 - accuracy: 0.9564
    Epoch 3/5
    1875/1875 [==============================] - 2s 976us/step - loss: 0.1086 - accuracy: 0.9669
    Epoch 4/5
    1875/1875 [==============================] - 2s 939us/step - loss: 0.0877 - accuracy: 0.9733
    Epoch 5/5
    1875/1875 [==============================] - 2s 955us/step - loss: 0.0748 - accuracy: 0.9765





    <tensorflow.python.keras.callbacks.History at 0x1990fb92070>



`Model.evaluate` - check performance on a validation or test set


```python
model.evaluate(X_test, y_test, verbose=2)
```

    313/313 - 0s - loss: 0.0752 - accuracy: 0.9771





    [0.07517100125551224, 0.9771000146865845]


