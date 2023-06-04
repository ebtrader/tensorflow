# https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/25_august_2021_fixes/C1/W2/ungraded_labs/C1_W2_Lab_1_beyond_hello_world.ipynb

# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W2/ungraded_labs/C1_W2_Lab_1_beyond_hello_world.ipynb


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

np.set_printoptions(linewidth=200)

index_number = 101

plt.imshow(training_images[index_number])
print(training_labels[index_number])
print(training_images[index_number])

# make sure to add this to the end
plt.show() # https://stackoverflow.com/questions/42812230/why-plt-imshow-doesnt-display-the-image

# normalize it by / 255
training_images  = training_images / 255.0
test_images = test_images / 255.0

# design the model

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# build the model

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

