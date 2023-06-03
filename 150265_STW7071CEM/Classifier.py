import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import ClassifierChain
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline

train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

abstract_train_data = []
abstract_test_data = []
p_stemmer = PorterStemmer()
stop_words = stopwords.words('english')


# Remove StopWords and Stemming
def remove_stopwords(data=None):
    if data is None:
        data = []
    data_list = []
    for name in data:
        words = word_tokenize(name)
        stem_word = ""
        for word in words:
            if word.lower() not in stop_words:
                stem_word += p_stemmer.stem(word) + ' '
        data_list.append(stem_word.lower())
    return data_list


# Remove Special Characters
def remove_special_character(data=None):
    if data is None:
        data = []
    abstract_set = []
    special_characters = '''!()-—[]{};:'"\, <>./?@#$%^&*_~0123456789+=’‘'''
    for file in data:
        word_nps = ""
        if len(file.split()) == 1:
            abstract_set.append(file)
        else:
            for a in file:
                if a in special_characters:
                    word_nps += ' '
                else:
                    word_nps += a
            abstract_set.append(word_nps)
    return abstract_set


# Remove stopwords from Train Data
data_train = np.array(train_data['ABSTRACT'])
abstract_list_train = remove_stopwords(data_train)

# Remove stopwords from Test Data
data_test = np.array(test_data['ABSTRACT'])
abstract_list_test = remove_stopwords(data_test)

# Removing special characters from Train Data and Test Data
train_data_x = remove_special_character(abstract_list_train)
test_data_x = remove_special_character(abstract_list_test)

categories = ['Computer Science', 'Physics', 'Mathematics', 'Statistics']

train_data_y = train_data[categories]
test_data_y = test_data[categories]

print("There are ", len(train_data_x), " input training samples")
print("There are ", len(test_data_x), " input testing samples")
print("There are ", train_data_y.shape, " output training samples")
print("There are ", test_data_y.shape, " output testing samples")

# defining parameters for pipeline
parameters = Pipeline([('tfidf', TfidfVectorizer(stop_words=stop_words)), ('clf', ClassifierChain(MultinomialNB()))])

# train data
parameters.fit(train_data_x, train_data_y)

# predict
predictions = parameters.predict(test_data_x)

print('Accuracy = ', accuracy_score(test_data_y, predictions))
print('F1 score is ', f1_score(test_data_y, predictions, average="micro"))
print(classification_report(test_data_y, predictions))

# Confusion Matrix and HeatMap Generation
y_test_data = np.array(test_data_y.values.argmax(axis=1))
prediction_data = np.array(predictions.argmax(axis=1))
mat = confusion_matrix(y_test_data, prediction_data)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

with open('model_MultiNB.pkl', 'wb') as pickle_file:
    pickle.dump(parameters.named_steps['clf'], pickle_file)

