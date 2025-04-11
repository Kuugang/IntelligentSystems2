---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import tensorflow as tf
fro; tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```

```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```

```python
print("Training set shape: ", X_train.shape)
print("Testing set shape: ", X_test.shape)
```

```python
plt.figure(figsize = (10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i])
    plt.axis("off")
plt.show()
```

```python
# Normalize image data
X_test = X_test / 255
X_train = X_train / 255
```

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape = (32, 32, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32, (3,3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
```

```python
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))
```

```python
loss, accuracy = model.evaluate(X_test, y_test)
```

```python
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis = 1)
```

```python
labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer","Dog", "Frog", "Horse", "Ship", "Truck"]
```

```python
import random
index = random.randint(0, len(X_test) - 1)
plt.imshow(X_test[index])
plt.title(f"Predicted: {labels[predicted_labels[index]]}")
plt.axis("off")
plt.show()
```

```python
print(1)
```
