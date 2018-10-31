import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

def evaluate_models(X, y):
    # Creating dict to save scores to evaluate:
    f1_scores['MLPClassifier'] = []
    roc_scores['MLPClassifier'] = []
    f1_scores['RandomForestClassifier'] = []
    roc_scores['RandomForestClassifier'] = []

    # Initializing Stratified Cross Validation:
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Create a dic to save all metric results and print the average...
        perform_models(
            [
                MLPClassifier(solver='lbfgs'),
                RandomForestClassifier(n_estimators=100, n_jobs=-1),
            ],
            X_train, X_test,
            y_train, y_test
        )
    for model in f1_scores.keys():
        print(model + ' has f1 average: ' + str( sum(f1_scores[model]) / len(f1_scores[model]) ))
        print(model + ' has roc_auc average: ' + str( sum(roc_scores[model]) / len(roc_scores[model]) ))

# Function to perform a list of models:
def perform_models(classifiers, X_train, X_test, y_train, y_test):
    string = ''

    for classifier in classifiers:
        string += classifier.__class__.__name__

        # Train
        classifier.fit(X_train, y_train)
        # Predicting values with model:
        predicteds = classifier.predict(X_test)
        # Getting score metrics:
        f1 = f1_score(y_test, predicteds)
        roc = roc_auc_score(y_test, predicteds, average='weighted')

        # Adding scores:
        f1_scores[classifier.__class__.__name__].append(f1)
        roc_scores[classifier.__class__.__name__].append(roc)

        string += ' has f1: ' + str(f1) + ' roc_auc: ' + str(roc)+ '\n'
        print(string)
        string = ''


# Reading the dataset
card_transactions = pd.read_csv('../creditcard.csv')
'''
# Saving in a pickle to be read faster
#card_transactions.to_pickle('card_transactions.pkl')

#card_transactions = pd.read_pickle('card_transactions.pkl')

## 1. Understand the problem and the data
# Seeing the features
#print(card_transactions.shape)

#print(card_transactions.columns)

#print(card_transactions.head(3))

# See Class distribution
count_classes = pd.value_counts(card_transactions['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Data Class distribution")

plt.xlabel("Class")

plt.ylabel("Frequency")

## 2. Data exploration / data cleaning / data preprocessing


# Have any null value in the DataFrame?
#print(card_transactions.isnull().values.any())

# Determine the number of fraud and valid transactions in the entire dataset.

fraud_data = card_transactions[card_transactions['Class'] == 1]
normal_data = card_transactions[card_transactions['Class'] == 0]

print(fraud_data.shape)
print(normal_data.shape)

# How different are the amount of money used in different transaction classes?

print(fraud_data.Amount.describe())
print(normal_data.Amount.describe())

#FIND GOOD PLOT TO THIS AMOUNTS

## 3. Feature engineering / feature selection

# I am not going to perform feature engineering or feature selection in first instance.
# The dataset already has been downgraded in order to contain 30 features (28 anonamised + time + amount).
# As the description, they used PCA.
# The only thing I'm going to do is normalize the _Amount_. As we could see previously, have a lot of variantion on data.
'''

amount_values = card_transactions['Amount'].values
standardized_amount = StandardScaler().fit_transform(amount_values.reshape(-1, 1))
card_transactions['normAmount'] = standardized_amount
card_transactions = card_transactions.drop(['Time', 'Amount'], axis=1)
#print(card_transactions['normAmount'].head())

## 4. Model evaluation and selection
#1) Select Classifiers Algorithms to be used.
#2) Compare what happens when using resampling techniques and when not using it.
#3) Evaluate the models by using *Stratified Cross Validation (for not resampled), normal Cross Validation (for resampled) and some of the performance metrics mentioned before.
#4) Repeat the best resampling/not resampling method, by tuning the parameters.
# *Stratified Cross Validation is a recommended CV technique to large imbalance in the distribution of the target class which the folds are made by preserving the percentage of samples for each class.

X = card_transactions.iloc[:, card_transactions.columns != 'Class']
y = card_transactions.iloc[:, card_transactions.columns == 'Class']
f1_scores = {}
roc_scores = {}

evaluate_models(X, y)
## 5. Model optimization

## 6. Interpretation of results and predictions