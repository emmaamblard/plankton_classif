import tensorflow_tricks  # settings for tensorflow to behave nicely

from os import makedirs
from os.path import dirname, join, isdir
import argparse

import pandas as pd
import numpy as np

from importlib import reload
import dataset            # custom data generator
import cnn                # custom functions for CNN generation
import biol_metrics       # custom functions model evaluation
dataset = reload(dataset)
cnn = reload(cnn)
biol_metrics = reload(biol_metrics)


# options to display all rows and columns for large DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


PATH = dirname(__file__)
DEFAULT_DATA_PATH = join(PATH, '../io/data/')
DEFAULT_SAVE_PATH = join(PATH, '../io/')
DEFAULT_MODEL_PATH = join(DEFAULT_SAVE_PATH, 'models/')

parser = argparse.ArgumentParser(description='Script to train a model')

parser.add_argument('--dataset', type=str, default='debug', help='Name of the dataset to train on')
parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to data folder')
parser.add_argument('--model_name', type=str, default='debug', help='Model name')
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path where to save the model and its checkpoints')
parser.add_argument('--model', type=str, default='mobilenet_v2_140_224', help='Model architecture to use')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--non_biol_classes', type=str, default='', help='Labels of non biological classes separated by commas for evaluation')
parser.add_argument('--fc_layer_size', type=str, default="600", help='size of the last(s) fc layer(s)') 

args = parser.parse_args()

dataset_name = args.dataset
data_path = args.data_path
model_name = args.model_name
model_path = join(args.model_path, model_name)
model_type = args.model
non_biol_classes = [] if args.non_biol_classes == '' else args.non_biol_classes.split(',')


print('Set options')

# directory to save training checkpoints
ckpt_dir = join(model_path, 'checkpoints/')
makedirs(ckpt_dir, exist_ok=True)

# options for data generator (see dataset.EcoTaxaGenerator)
batch_size = args.batch_size
augment = True
upscale = True
with open(join(data_path, 'crop.txt')) as f:
    bottom_crop = int(f.read())

# CNN structure (see cnn.Create and cnn.Compile)
model_handle_map = {
    "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
    "efficientnet_v2_S": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
    "efficientnet_v2_XL": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"
}
fe_url = model_handle_map[model_type]
input_shape = (224, 224, 3)
fe_trainable = True
fc_layers_sizes = [int(x) for x in args.fc_layer_size.split(',')]
fc_layers_dropout = 0.4
classif_layer_dropout = 0.2

# CNN training (see cnn.Train)
use_class_weight = True
weight_sensitivity = 0.25  # 0.5 = sqrt
lr_method = 'decay'
initial_lr = 0.0005
decay_rate = 0.97
loss = 'cce'
epochs = args.epochs
workers = 10

print('Prepare datasets')

# read DataFrame with image ids, paths and labels
train_csv_path = join(join(data_path, dataset_name), 'train_labels.csv')
val_csv_path = join(join(data_path, dataset_name), 'valid_labels.csv')
test_csv_path = join(join(data_path, dataset_name), 'test_labels.csv')
df_train = pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)
df_test = pd.read_csv(test_csv_path)

# count nb of examples per class in the training set
class_counts = df_train.groupby('label').size()
class_counts

# list classes
classes = class_counts.index.to_list()

# generate categories weights
# i.e. a dict with format { class number : class weight }
if use_class_weight:
    max_count = np.max(class_counts)
    class_weights = {}
    for idx,count in enumerate(class_counts.items()):
        class_weights.update({idx : (max_count / count[1])**weight_sensitivity})
else:
    class_weights = None

# define numnber of  classes to train on
nb_of_classes = len(classes)
    
# define data generators
train_batches = dataset.EcoTaxaGenerator(
    images_paths=df_train['img_path'].values,
    input_shape=input_shape,
    labels=df_train['label'].values, classes=classes,
    batch_size=batch_size, augment=augment, shuffle=True,
    crop=[0,0,bottom_crop,0])

val_batches = dataset.EcoTaxaGenerator(
    images_paths=df_val['img_path'].values,
    input_shape=input_shape,
    labels=df_val['label'].values, classes=classes,
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])
# NB: do not shuffle or augment data for validation, it is useless
    
test_batches = dataset.EcoTaxaGenerator(
    images_paths=df_test['img_path'].values,
    input_shape=input_shape,
    labels=None, classes=None,
    batch_size=batch_size, augment=False, shuffle=False,
    crop=[0,0,bottom_crop,0])

print('Prepare model')

# try loading the model from a previous training checkpoint
my_cnn, initial_epoch = cnn.Load(ckpt_dir)

# if nothing is loaded this means the model was never trained
# in this case, define it
if (my_cnn is not None) :
    print('  restart from model trained until epoch ' + str(initial_epoch))
else :
    print('  define model')
    # define CNN
    my_cnn = cnn.Create(
        # feature extractor
        fe_url=fe_url,
        input_shape=input_shape,
        fe_trainable=fe_trainable,
        # fully connected layer(s)
        fc_layers_sizes=fc_layers_sizes,
        fc_layers_dropout=fc_layers_dropout,
        # classification layer
        classif_layer_size=nb_of_classes,
        classif_layer_dropout=classif_layer_dropout
    )

    print('  compile model')
    # compile CNN
    my_cnn = cnn.Compile(
        my_cnn,
        initial_lr=initial_lr,
        lr_method=lr_method,
        decay_steps=len(train_batches),
        decay_rate=decay_rate,
        loss=loss
    )

print('Train model') ## ----

# train CNN
history = cnn.Train(
    model=my_cnn,
    train_batches=train_batches,
    valid_batches=val_batches,
    epochs=epochs,
    initial_epoch=initial_epoch,
    log_frequency=1,
    class_weight=class_weights,
    output_dir=ckpt_dir,
    workers=workers
)

print('Evaluate model')

# load model for best epoch
best_epoch = None  # use None to get latest epoch
my_cnn, epoch = cnn.Load(ckpt_dir, epoch=best_epoch)
print(' at epoch {:d}'.format(epoch))

# predict classes for all dataset
pred, prob = cnn.Predict(
    model=my_cnn,
    batches=test_batches,
    classes=classes,
    workers=workers
)

# save prediction
eval_path = join(model_path, 'evaluation_results/')
if not isdir(eval_path):
    makedirs(eval_path)

df_test['predicted_label'] = pred
for i, label in enumerate(classes):
    df_test[label] = prob[:,i]
df_test.to_csv(join(eval_path, 'cnn_{}_predictions.csv'.format(model_name)))

# compute a few scores
cr = biol_metrics.classification_report(y_true=df_test.label, y_pred=df_test.predicted_label, y_prob=prob,
  non_biol_classes = non_biol_classes)
print(cr)
cr.to_csv(join(eval_path, 'cnn_{}_classification_report.csv'.format(model_name)))

# save model
my_cnn.save(model_path, include_optimizer=False)
# NB: do not include the optimizer state: (i) we don't need to retrain this final
#     model, (ii) it does not work with the native TF format anyhow.
