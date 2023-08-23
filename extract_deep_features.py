import tensorflow_tricks  # settings for tensorflow to behave nicely

from os import makedirs
from os.path import dirname, join
import argparse

import pickle

import pandas as pd
import tensorflow as tf

import dataset   # custom data generator


PATH = dirname(__file__)
DEFAULT_DATA_PATH = join(PATH, '../io/data/')
DEFAULT_SAVE_PATH = join(PATH, '../io/')
DEFAULT_MODEL_PATH = join(DEFAULT_SAVE_PATH, 'models/')

parser = argparse.ArgumentParser(description='Script to extract deep features with PCA')

parser.add_argument('--dataset', type=str, default='debug', help='Name of the dataset to train on')
parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to data folder')
parser.add_argument('--source', type=str, default='test', help='Set on which to extract features (train, valid or test)')
parser.add_argument('--model_name', type=str, default='model', help='Model name')
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the model folder containing the feature extractor')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--mode', type=str, default='default', help='Which type of feature extractor to use: resize or default')
parser.add_argument('--no_pca', action='store_true', dest='no_pca', default=False, help='Option to not perform PCA, False by default')
parser.add_argument('--folder_out', action='store', dest='folder_out', type=str, default='', help='The path to save where to save results')

args = parser.parse_args()

dataset_name = args.dataset
data_path = args.data_path
model_name = args.model_name
model_path = join(args.model_path, model_name)

# Create output directory
folder_out = args.folder_out
if folder_out:
    makedirs(folder_out, exist_ok=True)

print('Set options')

batch_size = args.batch_size  # size of images batches in GPU memory
workers = 10                  # number of parallel threads to prepare batches
with open(join(data_path, 'crop.txt')) as f:
    bottom_crop = int(f.read())

print('Load feature extractor and dimensionality reducer')

fe_name = 'feature_extractor' if args.mode == 'default' else 'feature_extractor_{}'.format(args.mode)
my_fe = tf.keras.models.load_model(join(model_path, fe_name), compile=False)
# get model input shape
input_shape = my_fe.layers[0].input_shape
# remove the None element at the start (which is where the batch size goes)
input_shape = tuple(x for x in input_shape if x is not None)

print('Load data and extract features')

# read DataFrame with image ids, paths and labels
# NB: those would be in the database in EcoTaxa
df = pd.read_csv(join(join(data_path, dataset_name), '{}_labels.csv'.format(args.source)))

# prepare data batches
batches = dataset.EcoTaxaGenerator(
    images_paths=df.img_path.values,
    input_shape=input_shape,
    labels=None, classes=None,
    # NB: we don't need the labels here, we just run images through the network
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])

# extract features by going through the batches
deep_features = my_fe.predict(batches, max_queue_size=max(10, workers*2), workers=workers)
deep_features_name = '{}_deep_features'.format(args.source)
deep_features_name = deep_features_name if args.mode == 'default' else deep_features_name + '_{}'.format(args.mode)

if not args.no_pca:
    #reduce their dimension
    with open(join(model_path, 'dim_reducer.pickle'),'rb') as pca_file:
        pca = pickle.load(pca_file)
    deep_features = pca.transform(deep_features)
else:
    deep_features_name += '_no_pca'

# save them to disk
deep_features_df = pd.DataFrame(deep_features, index=df.objid)
deep_features_df.reset_index(inplace=True)
if folder_out:
    deep_features_df.to_csv(join(folder_out, '{}.csv'.format(deep_features_name)), index=False)
else:
    deep_features_df.to_csv(join(model_path, '{}.csv'.format(deep_features_name)), index=False)
