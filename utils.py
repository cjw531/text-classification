import re, string
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import pandas as pd

''' EDA Functions '''
def word_count(df: pd.DataFrame) -> plt:
    ''' Plot word count histogram for each label '''
    label_len = len(df['target'].unique())
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    label_idx = 0
    fig, axs = plt.subplots(1, label_len)
    axs = np.array(axs)
    for ax in axs.reshape(-1):
        train_words = df[df['target'] == label_idx]['word_count']
        ax.hist(train_words)
        avg = df[df['target'] == label_idx]['word_count'].mean()
        ax.set_title('Label: ' + str(label_idx) + '\n(Avg: ' + str(round(avg, 2)) + ')')
        label_idx += 1
    fig.suptitle('Word Count Histogram')
    fig.tight_layout()

    return plt

def char_count(df: pd.DataFrame) -> plt:
    ''' Plot char count histogram for each label '''
    label_len = len(df['target'].unique())
    df['char_count'] = df['text'].apply(lambda x: len(str(x)))

    label_idx = 0
    fig, axs = plt.subplots(1, label_len)
    axs = np.array(axs)
    for ax in axs.reshape(-1):
        train_words = df[df['target'] == label_idx]['char_count']
        ax.hist(train_words)
        avg = df[df['target'] == label_idx]['char_count'].mean()
        ax.set_title('Label: ' + str(label_idx) + '\n(Avg: ' + str(round(avg, 2)) + ')')
        label_idx += 1
    fig.suptitle('Character Count Histogram')
    fig.tight_layout()

    return plt

def unique_word_count(df: pd.DataFrame) -> plt:
    ''' Plot unique word count histogram for each label '''
    label_len = len(df['target'].unique())
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).split())))

    label_idx = 0
    fig, axs = plt.subplots(1, label_len)
    axs = np.array(axs)
    for ax in axs.reshape(-1):
        train_words = df[df['target'] == label_idx]['unique_word_count']
        ax.hist(train_words)
        avg = df[df['target'] == label_idx]['unique_word_count'].mean()
        ax.set_title('Label: ' + str(label_idx) + '\n(Avg: ' + str(round(avg, 2)) + ')')
        label_idx += 1
    fig.suptitle('Unique Word Count Histogram')
    fig.tight_layout()

    return plt

def lemmatizer(string: str) -> str:
    ''' Tokenize input sentence '''
    wl = WordNetLemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # map the position tag and lemmatize the word/token
    return " ".join(a)

def get_wordnet_pos(tag: str) -> wordnet:
    ''' nltk position helper '''
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def stopword(string: str) -> str:
    ''' Remove stopwords '''
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

def preprocess(text: str) -> str:
    ''' Convert to lowercase, strip and remove punctuations '''
    text = text.lower() 
    text = text.strip()
    text = re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

def text_preprocess(string: str) -> str:
    ''' Text preprocessing wrapper function '''
    return lemmatizer(stopword(preprocess(string)))

def get_max_words(df:pd.DataFrame) -> int:
    ''' Get maximum number of words from the dataset (column from dataframe) '''
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    max_words = df["word_count"].max()
    return max_words

def encode_label(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ''' Encode label and make it into integer '''
    le = LabelEncoder()
    encoded = le.fit_transform(np.ravel(df['target']))
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    df = df.rename(columns={'target': 'target_old'})
    df['target'] = encoded.tolist()
    return df, le_name_mapping

def draw_matrix(y_test, y_pred, title: str, labels: list[str]) -> plt:
    ''' Plot confusion matrix '''
    cm = confusion_matrix(y_test, y_pred)
    figure, ax = plot_confusion_matrix(conf_mat = cm, class_names = labels, show_absolute = False, show_normed = True, colorbar = True)
    plt.title(title)
    return plt
    
def sklearn_roc_curve(y_test, y_probs, names: list[str]) -> plt:
    ''' 
    Plot ROC curve for binary classifiers
    '''
    init_disp = None
    for i in range(len(y_probs)):
        if i == 0: init_disp = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_probs[i], name=names[i])
        else: disp = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_probs[i], name=names[i], ax=init_disp.ax_)
    plt.legend(fontsize=11)
    return plt

def gpu_management():
    ''' Set up GPU for tensorflow-based models, modify as needed '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                # limit the VRAM memory if required (set to 2GB for now)
                tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
            