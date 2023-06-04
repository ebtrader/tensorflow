import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
# path = f"{getcwd()}/../tmp2/mnist.npz"


# GRADED FUNCTION: train_mnist
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    # if logs is None:
    #     logs = {}
    if(logs.get('accuracy')>0.6):
        print("\nReached 60% accuracy so cancelling training!")
        self.model.stop_training = True

def train_mnist():
    mnist = tf.keras.datasets.mnist

#     (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    (x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(x_train, y_train, epochs=5, callbacks=[callbacks])
    # model fitting
    print(history.epoch, history.history['accuracy'][-1])


train_mnist()

