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
