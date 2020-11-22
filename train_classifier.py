import sys

# import libraries
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    df = pd.read_sql('MessagesML',database_filepath)
    X = df["message"].values
    Y = df.drop(['id', 'message','original','genre','Categories_sum','Categories_max'], axis=1)
    category_names = Y.columns.tolist()
    Y = df.drop(['id', 'message','original','genre','Categories_sum','Categories_max'], axis=1).values
    return X,Y,category_names

def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
    # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
   pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
    
            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    # specify parameters for grid search
   parameters = {
       'clf__estimator__n_neighbors': [20, 23, 25]
   }

    # create grid search object
   cv = GridSearchCV(pipeline, param_grid=parameters)
    
   return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print('Model test terminated')
    for i in range(len(category_names)):
        print(category_names[i], 'results:')
        print('F1 score:')
        print(f1_score(Y_test[:,i], y_pred[:,i], average= 'micro'))
        print('Recall score:')
        print(recall_score(Y_test[:,i], y_pred[:,i],average='micro'))
        print('Precision score:')
        print(precision_score(Y_test[:,i], y_pred[:,i],average='micro'))
    print("\nBest Parameters:", model.best_params_)    
        
        
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
    
        
        print('Training model...')
        model.fit(X_train, Y_train)
    
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()