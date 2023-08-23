import argparse
import json
import time
import pickle

from os import makedirs
from os.path import join

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from biol_metrics import classification_report


# options to display all rows and columns for large DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


DEFAULT_OUTPUT_PATH = "."
DEFAULT_SAVE_PATH = join(DEFAULT_OUTPUT_PATH, "grid_search_classifier_results")

IGNORE_FEATURES_DICT = {
    'ifcb': [],
    'flowcam': [],
    'isiis': ['cnn_score', 'label', 'avi_file'],
    'zoocam': ['_item_1', '_area_1', 'cutted20', 'isduplicate', 'param2', 'param3', 'param4', 'param5', 'label',
               'foundinframe', 'xmin', 'ymin', 'xmax', 'ymax', 'param1', 'bord_1', 'framename', 'annotation'],
    'zooscan': ['perim_', 'circ_', '_area', '_area_1', 'perim__1', 'circ__1', '_area_2'],
    'uvp6': ['xmg5', 'ymg5', 'compentropy', 'compmean', 'compslope', 'compm1', 'compm2', 'compm3', 'areai', 'esd',
             'centroids', 'cdexc', 'rawvig']
}

NON_BIOL_CLASSES_DICT = {
    'ifcb': ["other_living", "detritus", "other_living_elongated", "bad", "other_interaction", "spore"],
    'flowcam': ["dark", "detritus", "light", "lightsphere", "fiber", "lightrods", "contrasted_blob",
                "artefact", "darksphere", "darkrods", "ball_bearing_like", "other_living", "crumple sphere",
                "badfocus", "bubble", "dinophyceae_shape", "transparent_u"],
    'isiis': ["detritus", "streak", "other_living", "vertical line"],
    'zoocam': ["detritus", "fiber_detritus", "bubble", "light_detritus", "other_living", "artefact",
               "other_plastic", "medium_detritus", "gelatinous", "feces", "fiber_plastic"],
    'zooscan': ["detritus", "artefact", "fiber", "badfocus", "bubble", "other_egg", "seaweed", "Insecta", "other_living"],
    'uvp6': ["detritus", "fiber", "artefact", "reflection", "other<living", "dead<house", "darksphere"]
}

DEFAULT_RF_PARAMS = {
    "weight_sensitivity": 1.0,
    "random_state": 3,
    "n_jobs": 10
}


def get_native_features_from_csv(path, ignore_features=[], set="train", verbose=False):
    """
    Get labels and native features associated to a LOV dataset from a csv file

    Parameters:
        path: path to the folder containing the "<set>_labels.csv" file
        ignore_features: list of features to ignore when retrieving data
        set: the chosen split between "train", "val" and "test"
        verbose: verbose mode

    Returns:
        features_df: a pandas DataFrame containing the label and features for each object
    """
    # get all data from the csv file for a given set
    print("Get native features from csv", flush=True)
    csv_path = join(path, '{}_labels.csv'.format(set))
    features_df = pd.read_csv(csv_path, index_col="objid")

    # reformat dataframe
    features_df = features_df.drop(ignore_features + ["img_path"], axis=1)
    feature_columns = ['feature_{}'.format(i) for i in range(len(features_df.columns)-1)]
    features_df.columns = ["label"] + feature_columns

    return features_df


def get_deep_features_from_csv(path, set="train", verbose=False):
    """
    Get deep features (descriptors) from a csv file for a given set

    Parameters:
        path: path to the folder containing the "<set>_labels.csv" file
        set: the chosen split between "train", "val" and "test"
        verbose: verbose mode

    Returns:
        descriptors_df: a pandas DataFrame containing the deep features for each object
    """
    # get data from the csv file for a given set
    print("Get deep features from csv", flush=True)
    csv_path = join(path, '{}_deep_features.csv'.format(set))
    descriptors_df = pd.read_csv(csv_path, index_col="objid")

    # reformat dataframe
    feature_columns = ['deep_feature_{}'.format(i) for i in range(len(descriptors_df.columns))]
    descriptors_df.columns = feature_columns
    return descriptors_df


def train_rf_classifier(params, data_dict):
    """
    Train a Random Forest classifier

    Parameters:
        params: a dictionary containing the model parameters
        data_dict: a dictionary containing all train, valid and test data

    Returns:
        model: a trained Random Forest classifier
    """
    print("Define and train Random Forest classifier for parameters:\n{}".format(params), flush=True)

    # get train features and labels
    df_train = data_dict["train"]
    train_features = df_train.drop(["label"], axis=1).values
    train_labels = df_train["label"].values

    # add default classifier parameters
    for param in DEFAULT_RF_PARAMS:
        if param not in params:
            params[param] = DEFAULT_RF_PARAMS[param]
    
    # compute class weights
    weight_sensitivity = params.pop("weight_sensitivity")
    class_counts = df_train.groupby('label').size()
    classes = class_counts.index.to_list()

    max_count = np.max(class_counts)
    class_weight = {}
    for idx, count in enumerate(class_counts.items()):
        class_weight.update({classes[idx] : (max_count / count[1])**weight_sensitivity})

    # define and train classifier
    since = time.time()
    model = RandomForestClassifier(class_weight=class_weight, **params)
    model.fit(X=train_features, y=train_labels)
    time_elapsed = time.time() - since  # get training time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)

    return model


def apply_classifier(model, data_dict, classes, non_biol_classes):
    """
    Apply a trained classifier to a test set

    Parameters:
        model: the classifier to evaluate
        data_dict: a dictionary containing all train, valid and test data

    Returns:
        df_predictions: a pandas DataFrame containing predicted labels on the test set with their probabilities
        cr: a classification report as a pandas DataFrame
    """
    print('Apply classifier', flush=True)

    # get test features and labels
    df_test = data_dict["test"]
    test_features = df_test.drop(["label"], axis=1).values
    test_labels = df_test["label"].values

    # prediction on test set
    since = time.time()
    probs = model.predict_proba(test_features)
    time_elapsed = time.time() - since  # get prediction time
    print('Prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)

    # extract highest score and corresponding label
    predicted_labels = np.array(classes)[np.argmax(probs, axis=1)]

    # store predicted labels and probabilities in a dataframe
    df_predictions = df_test.drop(df_test.columns[1:], axis=1)
    df_predictions['predicted_label'] = predicted_labels
    for i, label in enumerate(classes):
        df_predictions[label] = probs[:, i]

    # compute classification report
    cr = classification_report(y_true=test_labels, y_pred=predicted_labels, y_prob=probs,
                               non_biol_classes=non_biol_classes)

    print('\n{}\n'.format(cr), flush=True)

    return df_predictions, cr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to train RF classifiers using grid search")

    parser.add_argument("--dataset", action="store", type=str, default="",
                        help="Name of the dataset to retrieve native features from database")
    parser.add_argument("--features_csv", action="store", type=str, default="",
                        help="Path to the folder containing labels and native feature csv files")
    parser.add_argument("--deep_features_csv", action="store", type=str, default="",
                        help="Path to the folder containing deep feature csv files")
    parser.add_argument("--ignore_features", action="store", type=str, default="",
                        help="List of features to ignore when retrieving data")
    parser.add_argument("--no_native", action="store_true", default=False,
                        help="Option to not use native features")
    parser.add_argument("--parameters", action="store", type=str, default="",
                        help="Path to a json file containing parameters for the grid search")
    parser.add_argument("--non_biol_classes", action="store", type=str, default="",
                        help="Labels of non biological classes separated by commas for evaluation")
    parser.add_argument("--folder_out", action="store", type=str, default="",
                        help="The path to the folder where to save the models and evaluation results")
    parser.add_argument("--save_model", action="store_true", default=False,
                        help="Option to save the model")

    parser.add_argument("-v", "--verbose", action="store_true", help="activation of verbose mode")

    args = parser.parse_args()


    dataset_name = args.dataset
    native_features_path = args.features_csv
    deep_features_path = args.deep_features_csv
    
    if native_features_path == "":
        print("ERROR: --features_csv must be given to retrieve labels", flush=True)
        exit(1)

    # use at least one type of features
    no_native = args.no_native
    no_deep = deep_features_path == ""
    if no_native and no_deep:
        print("ERROR: At least one type of features must be used,"
              "if --no_native is given, then --deep_features_csv must be given", flush=True)
        exit(1)
    
    if args.ignore_features == '':
        ignore_features = IGNORE_FEATURES_DICT[dataset_name] if dataset_name in IGNORE_FEATURES_DICT else []
    else:
        ignore_features = [feature for feature in args.ignore_features.split(',')]

    parameters_path = args.parameters

    if parameters_path == "":
        print("ERROR: No parameters given\n"
              "please give the path to the file containing the grid search parameters using --parameters", flush=True)
        exit(1)

    if args.non_biol_classes == '':
        non_biol_classes = NON_BIOL_CLASSES_DICT[dataset_name] if dataset_name in NON_BIOL_CLASSES_DICT else []
    else:
        non_biol_classes = [label for label in args.non_biol_classes.split(',')]

    folder_out = args.folder_out
    if folder_out == "":
        print("WARNING: No folder_out path was given, "
              "by default all results will be saved in {}".format(DEFAULT_SAVE_PATH), flush=True)
        folder_out = DEFAULT_SAVE_PATH
    makedirs(folder_out, exist_ok=True)

    save_model = args.save_model
    verbose = args.verbose

    # get data for all sets (train, valid, test)
    data_dict = dict()
    splits = ["train", "valid", "test"]
    for split in splits:
        print("Read {} data".format(split), flush=True)
        
        df = get_native_features_from_csv(native_features_path, ignore_features, split, verbose)

        if no_native:
            print("We don't use native features", flush=True)
            feature_columns = [col for col in df.columns if col.startswith('feature')]
            df = df.drop(feature_columns, axis=1)

        if no_deep:
            print("We don't use deep features", flush=True)
        else:
            deep_features = get_deep_features_from_csv(deep_features_path, split, verbose)
            df = df.merge(deep_features, on="objid")

        if "photo_id" in df.columns:
            df = df.drop(["photo_id"], axis=1)
        
        data_dict[split] = df

    with open(parameters_path) as params_file:
        print("Read grid search parameters in {}".format(parameters_path), flush=True)
        parameters = json.load(params_file)
    parameter_grid = ParameterGrid(parameters)

    # create output folders
    pred_folder = join(folder_out, "predictions")
    eval_folder = join(folder_out, "evaluations")
    makedirs(pred_folder, exist_ok=True)
    makedirs(eval_folder, exist_ok=True)
    if save_model:
        models_folder = join(folder_out, "models")
        makedirs(models_folder, exist_ok=True)

    # grid search on parameters
    for params in parameter_grid:

        # prepare the name of saved files to identify different parameters combinations
        name_prefix = ""
        for param in params:
            for word in param.split('_'):
                name_prefix += word[0]
            name_prefix += "{}_".format(params[param])
        name_prefix = name_prefix[:-1]

        # train classifier
        model = train_rf_classifier(params, data_dict)
        classes = model.classes_

        # apply classifier
        df_predictions, cr = apply_classifier(model, data_dict, classes, non_biol_classes)

        print('Save results', flush=True)

        # save predictions
        df_predictions.to_csv(join(pred_folder, '{}_predictions.csv'.format(name_prefix)))

        # save classification report
        cr.to_csv(join(eval_folder, '{}_classification_report.csv'.format(name_prefix)))

        # save model
        if save_model:
            model_path = join(models_folder, 'rf_{}.pickle'.format(name_prefix))
            print("Saving model in {}".format(model_path), flush=True)
            with open(model_path, 'wb') as model_file:
                pickle.dump(model, model_file)

