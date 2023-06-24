import matplotlib.pyplot as plt
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import argparse

argparser = argparse.ArgumentParser(description='Set training parameters')
argparser.add_argument('-f', '--filename',
                       help='name of filename to load history')

args = argparser.parse_args()

# parse arguments
NAME = args.filename
# read loss values from CSV file
history = pd.read_csv(NAME)

loss_history = history['loss']
val_loss_history = history['val_loss']

# plot and save figure
plt.plot(loss_history)
plt.plot(val_loss_history)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()
