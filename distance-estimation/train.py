import os
import time
import argparse
import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

argparser = argparse.ArgumentParser(description='training the model')
argparser.add_argument('-r', '--results', help='output directory path')
argparser.add_argument('--train', help='train dataset path')
argparser.add_argument('--test', help='test dataset path')

args = argparser.parse_args()

# parse arguments
results_dir = args.results
train_dataset = args.train
test_dataset = args.test


def main():
    # import data and scaling
    df_train = pd.read_csv(train_dataset)
    df_test = pd.read_csv(test_dataset)

    x_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_train = df_train[['zloc']].values

    X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_test = df_test[['zloc']].values

    # standardized data
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    y_train = scalar.fit_transform(y_train)

    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # define callbacks
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=0, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    modelname = "model@{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(modelname))
    
    loss = []
    val_loss = []
    
    # start training
    history = model.fit(x_train, y_train, validation_split=0.2,
                        epochs=100, batch_size=32, callbacks=[tensorboard], verbose=1)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # save model and weights
    model_json = model.to_json()
    with open(os.path.join(results_dir, "{}.json".format(modelname)), "w") as json_file:
        json_file.write(model_json)

    model.save_weights(os.path.join(results_dir, "{}.h5".format(modelname)))
    print("Saved model to disk")

    # save model loss and validation loss 
    history_dict = {'loss': loss, 'val_loss': val_loss}
    df_history = pd.DataFrame(history_dict)
    df_history.to_csv(os.path.join(results_dir, '{}_results.csv'.format(modelname)))


if __name__ == '__main__':
    main()
