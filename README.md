# plankton_classif

Benchmark for plankton images classifications methods for images from multiple plankton imaging devices (IFCB, FlowCam, ISIIS, ZooCam, Zooscan, UVP6)

This code was used to run the comparison between multiple Convolutional Neural Network architectures and a Random Forest classifier in the paper Assessment of machine learning models for plankton image classification: emphasizing feature extraction over classifier complexity.

## Data

### Instruments

The comparison is to be done on data from multiple plankton imaging devices:

-   IFCB (Imaging FlowCytobot)
-   Flowcam
-   ISIIS (In Situ Ichthyoplankton Imaging System)
-   ZooCam
-   ZooScan
-   UVP (Underwater Vision Profiler)

### Input Data

Store your input data in `data/<instrument_name>` (by default, the `data` folder is in `../io/data`). Your data must contain an images folder with your images, as well as 3 csv files named `train_labels.csv`, `valid_labels.csv` and `test_labels.csv` with one row per object. These csv files should contain the following columns:

-   img_path: path to image
-   label: object classification
-   features_1 to features_n: object features for random forest fit (choices for names of these columns are up to you)

The `data/` folder should also contain a file named `crop.txt` containing the number of pixels to crop at the bottom of your images, in case there is a legend to remove (just input 0 if you don't need to crop your images).

It is strongly recommended that each class contain at least 100 images.

## Classification models

This code can be used to train two types of classification models: Convolutional Neural Networkd (CNN) and Random Forests (RF).

In both cases, training is done in two phases:

-   the model is optimized by training on the training set (defined by the `train_labels.csv` file in our case) and evaluating on the validation set (`valid_labels.csv`)
-   the optimized model is evaluated on the test set (`test_labels.csv`) never used before

### Convolutional Neural Network

#### Model description

A convolutional neural network takes an image as input and predicts a class for this image.

The CNN backbone can be : 
- a MobileNetV2 feature extractor (<https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4>)
- an EfficientNet V2 S feature extractor (<https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2>)
- an EfficientNet V2 XL feature extractor (<https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2>)

A classification head with the number of classes to predict is added on top of the backbone. Intermediate fully connected layers with customizable dropout rate can be inserted between both.

Input images are expected to have color values in the range [0,1] and a size of 224 x 224 pixels. If need be, images are automatically resized by the EcoTaxaGenerator.

#### CNN training

The CNN can be trained with `train_cnn.py`. For each step (i.e. epoch) in the training of the CNN model, the model is trained on training data and evaluated on validation data. Last saved weights are then used to test the model on the test data.

#### Output

After training, an output directory is created (by default `../io/models`) and results are stored in a subfolder with the model name.

The CNN can also be used as a deep features extractor with the following scripts:

-   `convert_to_feature_extractor.py` converts the model to a feature extractor
-   `train_dimensionality_reduction.py` trains a PCA to apply to the output of the feature extractor
-   `extract_deep_features.py` uses the feature extractor to extract deep features and optionnaly applies PCA.

All results of theses steps are saved in the same subfolder.

### Random Forest

#### Model description

A random forest takes a vector of features as input and predicts a class from these values.

#### RF training

Random Forests can be trained using `grid_search_rf.py`. Parameters are optimized with a gridsearch including:

-   number of trees
-   number of features to use to compute each split (default for classification is sqrt(n_features))
-   minimum number of samples required to be at a leaf node (default for classification is 5)

For each set of parameters, a model is trained on training data and evaluated on test data.

For the purpose of the paper, only the number of trees was searched for, similarly is fine-tuning the number of epochs when training a CNN.

#### Output

After training, an output directory is created (by default `./grid_search_classifier_results`) and results are stored in this folder for each RF trained with grid search.

## Results

After all trainings are performed, classification performance are assessed:

-   `classification_report.py` generates all classification reports as csv files into `./perf`
-   `detailed_metrics.py` computes all classification metrics and writes them into `./perf/prediction_metrics.csv`
-   `make_plots.R` reads the content of `./perf/prediction_metrics.csv` and generates figures for the paper, saved in `./figures`
