import tensorflow_tricks  # settings for tensorflow to behave nicely

from os.path import dirname, join
import argparse

import tensorflow as tf


PATH = dirname(__file__)
DEFAULT_MODEL_PATH = join(PATH, '../io/models/')

parser = argparse.ArgumentParser(description='Script to convert a trained model to a feature extractor')

parser.add_argument('--model_name', type=str, default='model', help='Model name')
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path where the model folder is')
parser.add_argument('--mode', type=str, default='default', help='Keep the last (resized) dense layer or not in the feature extractor')

args = parser.parse_args()

model_name = args.model_name
model_path = join(args.model_path, model_name)


# read model weights
my_cnn = tf.keras.models.load_model(model_path, compile=False)

if args.mode == 'default':
    print('default mode: We keep the original size of the model.')
    # drop the Dense and Dropout layers to get only the feature extractor
    my_fe = tf.keras.models.Sequential(
        [layer for layer in my_cnn.layers
         if not (isinstance(layer, tf.keras.layers.Dense) | 
                 isinstance(layer, tf.keras.layers.Dropout))
        ])

else:
    print('resize mode: We keep the second layer of the model, with a resize.')
    # drop the Dropout layers and the decision layer
    layers=[layer for layer in my_cnn.layers if not (isinstance(layer, tf.keras.layers.Dropout))]
    del(layers[-1])
    my_fe = tf.keras.models.Sequential(layers)

my_fe.summary()

# save feature extractor
fe_name = 'feature_extractor' if args.mode == 'default' else 'feature_extractor_{}'.format(args.mode)
my_fe.save(join(model_path, fe_name))
