# %% [markdown]
# # simple model

# %%
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# %% [markdown]
# # load data

# %%
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %%
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# %%
data = {
    "train_images": train_images,
    "test_images": test_images,
    "train_labes": train_labels,
    "test_labels": test_labels
}

# %% [markdown]
# ## type

# %%
for key in data:
    print(f"{key}: {type(data[key])}")

# %% [markdown]
# ## shape

# %%
for key in data:
    print(f"{key}: {data[key].shape}")

# %% [markdown]
# ## range

# %%
for key in data:
    print(f"{key}: {data[key].min()}, {data[key].max()}")

# %% [markdown]
# ## labels

# %%
np.unique(train_labels), np.unique(test_labels)

# %% [markdown]
# ## sample

# %%
# train_images[0]

# %% [markdown]
# ## look at data

# %%
plt.figure()
# plt.imshow(train_images[0])
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()

# %%
train_labels[0]

# %%
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# %% [markdown]
# # preprocess data

# %%
train_images_norm = train_images / 255.0
test_images_norm = test_images / 255.0

# %%
train_images_norm.min(), train_images_norm.max() 

# %%
test_images_norm.min(), test_images_norm.max() 

# %%
plt.figure()
# plt.imshow(train_images_norm[0])
plt.imshow(train_images_norm[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()

# %%
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images_norm[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# %% [markdown]
# # build the model

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10)
])


# %% [markdown]
# $$
#         \left[\begin{array}{ccc}
#             x_{1,1} & \cdots & x_{1,28} \\
#             x_{2,1} & \cdots & x_{2,28} \\
#             \vdots  & \ddots & \vdots \\
#             x_{28,1} & \cdots & x_{28,28} \\
#         \end{array}\right]
# $$

# %% [markdown]
# $$
# \left[ x_{1,1} \dots x_{1,28} x_{2,1} \dots x_{2,28} x_{28,1} \dots x_{28,28} \right]
# $$

# %% [markdown]
# $$
# \begin{align}
# &y_{1} = relu(w_{11}x_{1} + w_{12}x_{2} + \dots + w_{1,784}x_{784} + b_{1}) \\
# &y_{2} = relu(w_{21}x_{1} + w_{22}x_{2} + \dots + w_{2,784}x_{784} + b_{2}) \\
# &\vdots \\
# &y_{128} = relu(w_{128,1}x_{1} + w_{128,2}x_{2} + \dots + w_{128,784}x_{784} + b_{128})
# \end{align}
# $$

# %%
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
plt.ylim(-1, 5)
plt.grid()
plt.plot(x, relu(x))
plt.show()

# %% [markdown]
# $$
# \begin{align}
# &z_{1} = f(Y_{1}) = Y_{1} = w_{1,1}y_{1} + w_{1,2}y_{2} +\dots w_{1,128}y_{128} + b_{1}\\
# &z_{2} = f(Y_{2}) = Y_{2} = w_{2,1}y_{1} + w_{2,2}y_{2} +\dots w_{2,128}y_{128} + b_{2}\\
# &\vdots \\
# &z_{10} = f(Y_{10}) = Y_{10} = w_{10,1}y_{1} + w_{10,2}y_{2} +\dots w_{10, 128}y_{128} + b_{128}\\
# \end{align}
# $$

# %% [markdown]
# $$
#   z_{i} \geq 0 \\
#     and \\
#     \sum{z_{i}} = 1
# $$

# %%
model.summary()

# %% [markdown]
# # complie the model

# %%
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["sparse_categorical_accuracy"]
#     metrics=["accuracy"]
)

# %% [markdown]
# # fit

# %%
model.fit(
    train_images_norm,
    train_labels,
    epochs=5
)

# %% [markdown]
# # evaluate

# %%
model.evaluate(test_images_norm, test_labels, verbose=2)

# %% [markdown]
# # predict

# %%
raw_predictions = model.predict(test_images_norm)

# %%
raw_predictions.shape

# %%
raw_predictions[0]

# %%
raw_predictions[0].sum()

# %%
np.argmax(raw_predictions[0])

# %%
test_labels[0]

# %% [markdown]
# ## prob model

# %%
prob_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# %%
predictions = prob_model.predict(test_images_norm)

# %%
predictions.shape

# %%
predictions[0]

# %%
predictions[0].sum()

# %%
np.argmax(predictions[0])

# %%
np.argsort(-predictions[0])

# %%
np.argsort(-raw_predictions[0])

# %% [markdown]
# # verify model

# %%
plt.imshow(test_images_norm[0], cmap=plt.cm.binary)
plt.show()

# %%
plt.bar(range(10), predictions[0])
plt.xticks(range(10))
plt.show()

# %%
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(test_images_norm[0], cmap=plt.cm.binary)

plt.subplot(1,2,2)
plt.bar(range(10), predictions[0])
plt.xticks(range(10))

plt.show()


# %%
def verify_prediction(i):
    plt.figure(figsize=(6,3))

    plt.subplot(1,2,1)
    plt.imshow(test_images_norm[i], cmap=plt.cm.binary)

    plt.subplot(1,2,2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))

    plt.show()


# %%
verify_prediction(6)

# %% [markdown]
# # use the trained model

# %%
img = test_images_norm[0]

# %%
# prob_model.predict(img)

# %%
img.shape

# %%
img_3axis = np.expand_dims(img, 0)

# %%
img_3axis.shape

# %%
prob_model.predict(img_3axis)

# %%
predictions_single = prob_model.predict(img_3axis)

# %%
np.argmax(predictions_single)

# %% [markdown]
# # save and load the model

# %%
prob_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer="adam",
    metrics=["sparse_categorical_accuracy"]
#     metrics=["accuracy"]
)

prob_model.save("../saved_models/simple_model.h5")

# %%
loaded_model = tf.keras.models.load_model("../saved_models/simple_model.h5")

# %%
loaded_model.evaluate(test_images_norm, test_labels)

# %%
