import pandas as pd
import time
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

argparser = argparse.ArgumentParser(description='Continue training a model')
argparser.add_argument('-m', '--modelname',
                       help='model name (.json)')
argparser.add_argument('-w', '--weights',
                       help='weights filename (.h5)')

args = argparser.parse_args()

# parse arguments
MODEL = args.modelname
WEIGHTS = args.weights

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
y_train = df_train[['zloc']].values

# standardized data
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
y_train = scalar.fit_transform(y_train)

# define strategy
strategy = tf.distribute.MirroredStrategy()

# create model within the strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=4, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')

# load weights into new model
model.load_weights("{}".format(WEIGHTS))
print("Loaded model from disk")

modelname = "model@{}".format(int(time.time()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(modelname))

parallel_model = model
parallel_model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[tensorboard], verbose=1)

# ----------- save model and weights ----------- #
model_json = model.to_json()
with open("{}".format(modelname), "w") as json_file:
    json_file.write(model_json)

model.save_weights("{}".format(modelname))
print("Saved model to disk")
