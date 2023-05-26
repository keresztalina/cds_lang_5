##### IMPORT PACKAGES
# system tools
import os

# data munging tools
import pandas as pd
from joblib import dump, load

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics


##### DEFINE FUNCTIONS
def load_data(filename):

    # read csv
    data = pd.read_csv(
        filename)

    # extract needed columns from data frame
    X = data["clean_text"]
    y = data["is_depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, # inputs for the model
        y, # classification labels
        test_size = 0.1,   # create a 95/5 train/test split
        random_state = 42) # random state for reproducibility

    return X_train, X_test, y_train, y_test

def vectorize(X_train, X_test, vectorizer_path):

    # define vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range = (1, 2), # unigrams and bigrams (1 word and 2 word units)
        lowercase =  True, # don't distinguish between e.g. words at start vs middle of sentence
        max_df = 0.95, # remove very common words
        min_df = 0.05, # remove very rare words
        max_features = 500) # keep only top 500 features

    # first we fit the vectorizer to the training data...
    X_train_feats = vectorizer.fit_transform(X_train)

    #... then transform our test data
    X_test_feats = vectorizer.transform(X_test)

    dump(
        vectorizer, 
        vectorizer_path)

    return X_train_feats, X_test_feats

def classify(X_train_feats, X_test_feats, y_train, classifier_path):

    # define classifier
    classifier = MLPClassifier(
        activation = "relu", 
        hidden_layer_sizes = (10,), # 1 hidden layer of 20 neurons
        max_iter = 1000, # max number of attempts to converge
        early_stopping = True, # stop early if no improvement
        verbose = True, # print what's going on
        random_state = 42) # reproducibility

    # fit classifier
    classifier.fit(
        X_train_feats, 
        y_train)

    # get predictions
    y_pred = classifier.predict(
        X_test_feats)

    dump(
        classifier, 
        classifier_path)

    return y_pred

def evaluate(y_test, y_pred, eval_path):

    classifier_metrics = metrics.classification_report(
        y_test, 
        y_pred)

    with open(eval_path, 'w') as f:
        f.write(classifier_metrics)


##### DEFINE NECESSARY PATHS
# data input
in_path = os.path.join(
    "..",
    "data",  
    "dataset.csv")

# vectorizer output
vectorizer_path = os.path.join(
    "mdls",
    "vectorizer.joblib")

# classifier output
classifier_path = os.path.join(
    "mdls",
    "classifier.joblib")

# evaluation output
eval_path = os.path.join(
    "out",
    "metrics.txt")

##### MAIN
def main():

    # load data
    X_train, X_test, y_train, y_test = load_data(
        in_path)
    
    # vectorize & save vectorizer
    X_train_feats, X_test_feats = vectorize(
        X_train, 
        X_test,
        vectorizer_path)

    # classify & save classifier
    y_pred = classify(
        X_train_feats, 
        X_test_feats, 
        y_train,
        classifier_path)

    # evaluate & save evaluation
    evaluate(
        y_test,
        y_pred, 
        eval_path)

if __name__ == "__main__":
    main()





