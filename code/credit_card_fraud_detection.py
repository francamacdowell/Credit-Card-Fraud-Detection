import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
#import matplotlib.pyplot as plt
InteractiveShell.ast_node_interactivity = "all"

# Reading the dataset
card_transactions = pd.read_csv('../creditcard.csv')

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

## 4. Model evaluation and selection
#2 I will compare what happens when using resampling and when not using it. Will test this approach using a simple logistic regression classifier.
#3 I will evaluate the models by using some of the performance metrics mentioned above.
#4 I will repeat the best resampling/not resampling method, by tuning the parameters in the logistic regression classifier.
#5 I will finally perform classifications model using other classification algorithms.

## 5. Model optimization

## 6. Interpretation of results and predictions