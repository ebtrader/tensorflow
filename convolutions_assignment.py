import tensorflow as tf
from os import path, getcwd, chdir

DESIRED_ACCURACY = 0.6

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') is not None and logs.get('accuracy') >= DESIRED_ACCURACY): # change this to 'acc'
        print("\nReached 60% accuracy so cancelling training!")
        self.model.stop_training = True

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data('mnist.npz')

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

callbacks = myCallback()  # add this line for callback


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model fitting
history = model.fit(
    training_images, training_labels, epochs=5, callbacks=[callbacks]
)
# model fitting
print(history.epoch, history.history['accuracy'][-1]) # change this to 'acc' and use print instead of return
