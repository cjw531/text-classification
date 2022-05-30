# Performance Comparison of Binary and Multi Class Text Classification Models With `scikit-learn` and `TensorFlow`

Binary and multi class text classification models are proposed. Various metrics are also proposed along with the model, to compare their performances. 

`scikit-learn` Models:
- Logistic Regression
- Naive Bayes
- SVM (Support Vector Machine)
- Gradient Boosting

`TensorFlow` Models:
- CNN (Convolutional Neural Network) Model
- BERT Transformer Pre-Trained Model + CNN

<hr>

## TL;DR Quickstart
Refer to [`binary_classification.ipynb`](https://github.com/cjw531/text-classification/blob/main/binary_classification.ipynb) and [`multi_classification.ipynb`](https://github.com/cjw531/text-classification/blob/main/multi_classification.ipynb) for the demo and main runner flow.

<hr>

## Environment Setup

A conda environment setup file including all of the above dependencies are provided. Create the conda environment `textcls` by running:
```
conda env create -f environment.yml
```
For more informtaion, refer to [`environment.yml`](https://github.com/cjw531/text-classification/blob/main/environment.yml) file.

<hr>

## Dataset
The following datasets are used:
- Binary Dataset: [Disaster Tweets](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets) dataset (1: disaster Tweet; 0: not a disaster Tweet)
- Multi Class Dataset: [News Classification](https://www.kaggle.com/datasets/kishanyadav/inshort-news) dataset - only first 2 *.csv files are used(7 categorized classes in total: 'automobile', 'entertainment', 'politics', 'science', 'sports', 'technology', 'world')

All data should be saved under `./data/` subdirectory, with a `*.csv` format:
```
├── data
│   ├── inshort_news_data-1.csv
│   ├── inshort_news_data-2.csv
│   └── tweets.csv
```

Following cleanups are required:
- X data column name should be 'text' and y data (label) column name should be 'target'
- Label encoding should be also performed (e.g. Spam --> 1, Not Spam --> 0).
- In [`utils.py`](https://github.com/cjw531/text-classification/blob/main/utils.py), there is a helper method called `encode_label()` which does this task.

Sample data should be look like:
| text                           | target  |
|--------------------------------|:-------:|
| "this is a sample text"        |    0    |
| "this is another sample text"  |    1    |

Dataset also needs preprocessing. Tokenize input sentences, removing stop words, convert all char into lowercase, strip and removing punctuations are required. TF-IDF process is also required for the `sklearn` models. 

<hr>

## `scikit-learn` Text Classifiers
Models are modularized in [`model_sklearn.py`](https://github.com/cjw531/text-classification/blob/main/model_sklearn.py), so the model can be easily trained as follows:
```
lg_model = Logistic(X_train_vectors_tfidf, y_train, X_val_vectors_tfidf, y_test)
lg_model.runner()

nb_model = NaiveBayes(X_train_vectors_tfidf, y_train, X_val_vectors_tfidf, y_test)
nb_model.runner()

svm_model = SVM(X_train_vectors_tfidf, y_train, X_val_vectors_tfidf, y_test)
svm_model.runner()

gb_model = GradientBoosting(X_train_vectors_tfidf, y_train, X_val_vectors_tfidf, y_test)
gb_model.runner()

sklearn_roc_curve(y_test, [lg_model.y_prob, nb_model.y_prob, svm_model.y_prob, gb_model.y_prob], 
                        ['Logistic Regression', 'Naive Bayes', 'SVM', 'GradientBoosting']).show() # plotting roc curve
```

Following processes are performed within the `runner()` method:
1. Training with train dataset
2. Prediction with test dataset
3. Print report and confusion matrices
4. Provide runtime for training and testing
5. Provide CPU and system memory (RAM) usage

*Note that `sklearn` models do not utilize any GPU.

<hr>

## `TensorFlow` CNN Text Classifiers
This CNN model size is relatively smaller than the BERT + CNN model, and thus takes way much less time to train the model. Yet, it still reported the second best performance after the BERT + CNN model for both binary and multi class classification. Considering the tradeoff of the training time and the demand of resources (GPU), this CNN model can be one of the decent option to consider.

Models are modularized in [`model_cnn.py`](https://github.com/cjw531/text-classification/blob/main/model_cnn.py), so the model can be easily trained as follows:
```
max_words = get_max_words(df) # get maximum words based on the dataset
num_class = len(le_name_mapping) # number of classes

cnn_model = CNNBinary(X_train, y_train, X_test, y_test, max_words, num_class) # binary model
cnn_model.runner()

cnn_model = CNNMulti(X_train, y_train, X_test, y_test, max_words, num_class) # multi class model
cnn_model.runner()
```

Following processes are performed within the `runner()` method:
1. Training with train dataset
2. Prediction with test dataset
3. Print report and confusion matrices
4. Provide runtime for training and testing
5. Provide CPU and system memory (RAM) usage

*Note that GPU VRAM limit is set to 2GB, and you can change its limit in `gpu_management()` defined in [`utils.py`](https://github.com/cjw531/text-classification/blob/main/utils.py).

<hr>

## `TensorFlow` BERT + CNN Text Classifiers
With the benefit of the pretrained BERT model prior to the CNN layers, this BERT + CNN model revealed the best performance for both binary and multi class classification.

Models are modularized in [`model_bert.py`](https://github.com/cjw531/text-classification/blob/main/model_bert.py), so the model can be easily trained as follows:
```
num_class = len(le_name_mapping) # number of classes

bert_model = BERTBinary(X_train, y_train, X_test, y_test, num_class) # binary model
bert_model.runner()

bert_model = BERTMulti(X_train, y_train, X_test, y_test, num_class) # multi class model
bert_model.runner()
```

Following processes are performed within the `runner()` method:
1. Training with train dataset
2. Prediction with test dataset
3. Print report and confusion matrices
4. Provide runtime for training and testing
5. Provide CPU and system memory (RAM) usage

*Note that GPU VRAM limit is set to 2GB, and you can change its limit in `gpu_management()` defined in [`utils.py`](https://github.com/cjw531/text-classification/blob/main/utils.py).

<hr>

## System Specification
Note that the local system has the following configurations:
- CPU: AMD Ryzen&trade; 5 56000X 6-Core Processor
- RAM: TEAMGROUP T-FORCE 16GB x 2 DDR4 3600MHz
- GPU: NVIDIA<sup>&reg;</sup> GeForce RTX&trade; 3090 (24GB VRAM)

<hr>

## Future Work
1. Try data balancing or augmentation since the sample size is skewed especially for the disaster Tweet dataset.
2. Hyperparameter tuning especially with BYOD (i.e. batch size, epoch, and etc.)
3. Add/modify layers in CNN and the BERT transformer based model
4. Try MLP (Multi-Layer Perceptron) with and without BERT and compare the performance
