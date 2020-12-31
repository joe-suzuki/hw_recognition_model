# %% [markdown]
# # augmented model

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
# # preprocess data

# %%
train_images_norm = train_images / 255.0
test_images_norm = test_images / 255.0

# %%
train_images_norm.min(), train_images_norm.max() 

# %%
test_images_norm.min(), test_images_norm.max() 

# %% [markdown]
# # augment data

# %%
train_images_norm_4dim = np.expand_dims(train_images_norm, 3)

# %%
data_augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(
        factor=0.1,
        fill_mode="constant",
        fill_value=0
    ),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1,
        fill_mode="constant",
        fill_value=0
    ),
    tf.keras.layers.experimental.preprocessing.RandomZoom(
        height_factor=0.2,
        width_factor=0.2,
        fill_mode="constant",
        fill_value=0
    )
])

# %%
augmented_images = data_augmentation_layers(train_images_norm_4dim)
plt.figure(figsize=(10,10))
for i in range(30):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    if(i%2 == 0):
        plt.imshow(train_images_norm_4dim[i], cmap=plt.cm.binary)
        plt.xlabel("before")
    else:
        plt.imshow(augmented_images[i-1], cmap=plt.cm.binary)
        plt.xlabel("after")

# %% [markdown]
# # build the model

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10)
])

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
    augmented_images,
    train_labels,
    epochs=5
)

# %% [markdown]
# # evaluate

# %%
model.evaluate(test_images_norm, test_labels, verbose=2)

# %% [markdown]
# ## prob model

# %%
prob_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# %%
prob_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer="adam",
    metrics=["sparse_categorical_accuracy"]
#     metrics=["accuracy"]
)

# %%
prob_model.evaluate(test_images_norm, test_labels, verbose=2)

# %% [markdown]
# # save and load the model

# %%
prob_model.save("../saved_models/augmented_model.h5")

# %%
loaded_model = tf.keras.models.load_model("../saved_models/augmented_model.h5")

# %%
loaded_model.evaluate(test_images_norm, test_labels)

# %%
