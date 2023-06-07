# Importing libraries
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

from sklearn.svm import SVC, LinearSVC

# Read the clean dataset
reviews = pd.read_csv('SentimentModel/clean_train.csv')

X = reviews['content_stem']
y = reviews['target']

# Read the clean test dataset
test_reviews = pd.read_csv('SentimentModel/clean_test.csv')

X_test = test_reviews['content_stem']
y_test = test_reviews['target']

# Perform train test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Test split only
X_test1, X_test2, y_test1, y_test2 = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print('Creating TFIDF Naive Bayes Model')
# Create a pipeline with TF-IDF and Naive Bayes
pipe_tvec_nb = Pipeline([
    ('tvec', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# Search over the following values of hyperparameters:
pipe_tvec_nb_params = {
    'tvec__max_features': [500], #200
    'tvec__min_df': [2,3], #
    'tvec__max_df': [.9,.95], 
#     'tvec__ngram_range':[(1,1),(1,2)],  
}

# Instantiate GridSearchCV
gs_tvec_nb = GridSearchCV(pipe_tvec_nb, # Objects to optimise
                        param_grid = pipe_tvec_nb_params, # Hyperparameters for tuning
                        cv=6) # 10-fold cross validation

# Fit model on to training data
gs_tvec_nb.fit(X_train, y_train)

gs_tvec_nb_pred = gs_tvec_nb.predict(X_val)

gs_tvec_nb_pred_test = gs_tvec_nb.predict(X_test)

# Write a function that takes in the actual y value and model predictions, 
# and prints out the confusion matrix and classification report
# Dataset: Validation or test set

def cmat(actual_y, predictions, dataset):
    
    # Create a classification report
    print('Classification report for', dataset)
    print(classification_report(actual_y, predictions))
    print('')
    
    # Create a confusion matrix
    cm = confusion_matrix(actual_y, predictions)
    cm_df = pd.DataFrame(cm, columns=['Predicted Positive Review','Predicted Negative Review'], index=['Actual Positive Review', 'Actual Negative Review'])
    print('Confusion matrix for', dataset)
    print(cm_df)

print('Training score:', gs_tvec_nb.score(X_train, y_train))
print('Validation score:', gs_tvec_nb.score(X_val, y_val))
print('Accuracy: ', accuracy_score(y_val, gs_tvec_nb_pred))
print('')

# Print classification report and confusion matrix
print('\033[94m>gs_tvec_nb_pred\033[0m')
cmat(y_val, gs_tvec_nb_pred, 'validation set')
print('')
cmat(y_test, gs_tvec_nb_pred_test, 'TEST set')

print('Creating TFIDF SVC Model')
# Create a pipeline with TF-IDF Vectorizer and SVC
pipe_tvec_svc = Pipeline([
    ('tvec', TfidfVectorizer(stop_words='english')),
    ('linearsvc', LinearSVC(random_state=42)) 
])

# Search over the following values of hyperparameters:
pipe_tvec_svc_params = {
    'tvec__max_features': [500], #200,500
    'tvec__min_df': [2,3], 
    'tvec__max_df': [.9,.95], 
    'linearsvc__C': [.1]
}

# Instantiate GridSearchCV
gs_tvec_svc = GridSearchCV(pipe_tvec_svc, # Objects to optimise
                          param_grid = pipe_tvec_svc_params, # Hyperparameters for tuning
                          cv=6) # 10-fold cross validation

# Fit model on to training data
gs_tvec_svc.fit(X_train, y_train)

# Calibrate predicted probabilities using CalibratedClassifierCV
calibrated_lsvc = CalibratedClassifierCV(gs_tvec_svc.best_estimator_, cv=8)

# Fit the calibrated model on training data
calibrated_lsvc.fit(X_train, y_train)

# Predict class probabilities on validation set
y_proba = calibrated_lsvc.predict_proba(X_val)

# Generate predictions on validation set
tvec_svc_pred = calibrated_lsvc.predict(X_val)

# Print best parameters
print('Best parameters: ', gs_tvec_svc.best_params_)

# Print accuracy scores
print('Best CV score: ', gs_tvec_svc.best_score_)
print('Training score:', gs_tvec_svc.score(X_train, y_train))
print('Validation score:', gs_tvec_svc.score(X_val, y_val))
print('Accuracy: ', accuracy_score(y_val, tvec_svc_pred))
print('')

# Print classification report and confusion matrix
print('\033[94m>tvec_svc_pred\033[0m')
cmat(y_val, tvec_svc_pred, 'validation set')

print('Creating TFIDF LR Model')
# Create a pipeline with TF-IDF and Logistic Regression
pipe_tvec_lr = Pipeline([
    ('tvec', TfidfVectorizer(stop_words='english')),
    ('lr', LogisticRegression(random_state=42))
])

# Search over the following values of hyperparameters:
pipe_tvec_lr_params = {
    'tvec__max_features': [200], #100,200
    'tvec__min_df': [2,3], #2,3 
    'tvec__max_df': [.9,.95], 
#     'tvec__ngram_range':[(1,1),(1,2)],  
    'lr__penalty': ['l2'],
    'lr__C': [.1, 1] #.1, .01
}

# Instantiate GridSearchCV
gs_tvec_lr = GridSearchCV(pipe_tvec_lr, # Objects to optimise
                          param_grid = pipe_tvec_lr_params, # Hyperparameters for tuning
                          cv=10) # 10-fold cross validation

# Fit model on to training data
gs_tvec_lr.fit(X_train, y_train)

# Generate predictions on validation set
tvec_lr_pred = gs_tvec_lr.predict(X_val)

# Print best parameters
print('Best parameters: ', gs_tvec_lr.best_params_)

# Print accuracy scores
print('Best CV score: ', gs_tvec_lr.best_score_)
print('Training score:', gs_tvec_lr.score(X_train, y_train))
print('Validation score:', gs_tvec_lr.score(X_val, y_val))
print('Accuracy: ', accuracy_score(y_val, tvec_lr_pred))
print('')

# Print classification report and confusion matrix
print('\033[94m>tvec_lr_pred\033[0m')
cmat(y_val, tvec_lr_pred, 'validation set')

print('Creating Voting Classifier Model')
# Instantiate the Voting Classifier with TF-IDF Logistic Regression and SVC
voting_clf = VotingClassifier(
    estimators=[('tvec_nb', gs_tvec_nb),
                ('tvec_svc', calibrated_lsvc)], 
    voting='soft', 
    weights=[2,1]
)

# Fit model on to training data
voting_clf.fit(X_train, y_train)

# Generate predictions on validation set
voting_pred = voting_clf.predict(X_val)

# Generate predictions on test set
voting_pred_test = voting_clf.predict(X_test)

# Generate predictions on test set
voting_pred_fulltraining = voting_clf.predict(X)

# Print accuracy scores
print('Training score:', voting_clf.score(X_train, y_train))
print('Validation score:', voting_clf.score(X_val, y_val))
print('Accuracy: ', accuracy_score(y_val, voting_pred))
print('')

# Print classification report and confusion matrix
print('\033[94m>voting_pred\033[0m')
cmat(y_val, voting_pred, 'validation set')
print('')
cmat(y_test, voting_pred_test, 'TEST set')
print('')
cmat(y, voting_pred_fulltraining, 'FULL TRAINING set')

# cmat(y_test1, voting_pred, 'testing set')


print('Saving model to disk')
# Saving model to disk
pickle.dump(voting_clf, open('SentimentModel/modelCraigNumberOnly.pkl','wb'))

