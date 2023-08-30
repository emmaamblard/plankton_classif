import glob
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(description='Script to compute performance metrics.')
parser.add_argument('--predictions_path', type=str, help='Path to predictions')
args = parser.parse_args()

predictions_path = args.predictions_path

# Create dir to store reports
save_dir = 'classification_performance'
os.makedirs(save_dir, exist_ok=True)


## Get detailed and group classes for all datasets
# Initiate empty dataframe
df_all = pd.DataFrame()


# List datasets
datasets = ['flowcam', 'ifcb', 'isiis', 'uvp6', 'zoocam', 'zooscan']

sheet_id = '1C57EPnnOljtFKWrkdvQj2-UNc0bg0PMHb6km0YhaGSw'

for dataset in datasets:
    
    # Generate sheet url
    sheet_name = dataset
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    
    # Reed sheet
    df = pd.read_csv(url, usecols=['taxon', 'level2', 'plankton'])
    
    # Add name of dataset
    df['dataset'] = dataset
    
    # Store with other datasets
    df_all = pd.concat([df_all, df]).reset_index(drop = True)

# Rename grouped column
df_all = df_all.rename(columns={'level2': 'grouped'})


# Prepare computation of random perf
rand_comp = {
    'flowcam': False, 
    'ifcb': False, 
    'isiis': False, 
    'uvp6': False, 
    'zoocam': False, 
    'zooscan': False
}

## Compute grouped metrics 

## Deep models
# Initiate empty dict
metrics = {
    'dataset': [],
    'model': [],
    'accuracy': [],
    'balanced_accuracy': [],
    'plankton_precision': [],
    'plankton_recall': [],
    'accuracy_g': [],
    'balanced_accuracy_g': [],
    'plankton_precision_g': [],
    'plankton_recall_g': []
}

# List files of predictions
files = glob.glob(os.path.join(predictions_path, '*.csv'))
files.sort()


for file in files:
    # Read file
    df = pd.read_csv(file, usecols=['label', 'predicted_label'])
    
    # Get dataset and model name
    dataset = os.path.split(file)[1].split('_')[0]
    model = os.path.split(file)[1].split('_', 1)[1].replace('_predictions.csv', '')
    metrics['dataset'].append(dataset)
    metrics['model'].append(model)
    print(f'Processing {model} for {dataset}')
    
    # Get labels and predictions
    y = df['label']
    y_pred = df['predicted_label']
    
    # Space fix for zooscan dataset
    if dataset == 'zooscan':
        y = [e.replace('\u202f', '') for e in y]
        y_pred = [e.replace('\u202f', '') for e in y_pred]
    
    # Get classes for this dataset
    df_ref = df_all[df_all['dataset'] == dataset].reset_index(drop = True).drop('dataset', axis = 1)
    # Plankton classes
    plankton_classes = list(set(df_ref[df_ref['plankton']]['taxon'].tolist()))
    plankton_classes_g = list(set(df_ref[df_ref['plankton']]['grouped'].tolist()))
    
    # Compute grouped metrics
    metrics['accuracy'].append(accuracy_score(y, y_pred))
    metrics['balanced_accuracy'].append(balanced_accuracy_score(y, y_pred))
    metrics['plankton_precision'].append(precision_score(y, y_pred, labels=plankton_classes, average='weighted', zero_division=0))
    metrics['plankton_recall'].append(recall_score(y, y_pred, labels=plankton_classes, average='weighted', zero_division=0))
    
    # Generate dict for taxonomy match
    taxo_match = df_ref.set_index('taxon').to_dict('index')
    
    # Convert true classes to larger ecological classes
    y_g = np.array([taxo_match[t]['grouped'] for t in y])
    
    # Convert predicted classes to larger ecological classes
    y_pred_g = np.array([taxo_match[p]['grouped'] for p in y_pred])
    
    # Compute grouped metrics
    metrics['accuracy_g'].append(accuracy_score(y_g, y_pred_g))
    metrics['balanced_accuracy_g'].append(balanced_accuracy_score(y_g, y_pred_g))
    metrics['plankton_precision_g'].append(precision_score(y_g, y_pred_g, labels=plankton_classes_g, average='weighted', zero_division=0))
    metrics['plankton_recall_g'].append(recall_score(y_g, y_pred_g, labels=plankton_classes_g, average='weighted', zero_division=0))
    
    # Compute random perf if not done before
    if rand_comp[dataset] is False:
        
        # Shuffle true classes
        y_rand = shuffle(y, random_state = 12)
        metrics['dataset'].append(dataset)
        model = 'random'
        metrics['model'].append(model)
        print(f'Processing {model} for {dataset}')

        # Compute detailed metrics
        metrics['accuracy'].append(accuracy_score(y_rand, y_pred))
        metrics['balanced_accuracy'].append(balanced_accuracy_score(y_rand, y_pred))
        metrics['plankton_precision'].append(precision_score(y_rand, y_pred, labels=plankton_classes, average='weighted', zero_division=0))
        metrics['plankton_recall'].append(recall_score(y_rand, y_pred, labels=plankton_classes, average='weighted', zero_division=0))
        
        # Shuffle grouped true classes
        y_rand_g = shuffle(y_g, random_state = 12)

        # Compute detailed metrics
        metrics['accuracy_g'].append(accuracy_score(y_rand_g, y_pred_g))
        metrics['balanced_accuracy_g'].append(balanced_accuracy_score(y_rand_g, y_pred_g))
        metrics['plankton_precision_g'].append(precision_score(y_rand_g, y_pred_g, labels=plankton_classes, average='weighted', zero_division=0))
        metrics['plankton_recall_g'].append(recall_score(y_rand_g, y_pred_g, labels=plankton_classes, average='weighted', zero_division=0))
        
        # Tag random perf as done
        rand_comp[dataset] = True

# Convert to dataframe
metrics = pd.DataFrame(metrics)
print('Saving results')
metrics.to_csv('classification_performance/prediction_metrics.csv', index = False)


