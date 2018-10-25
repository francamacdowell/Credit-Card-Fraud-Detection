
# Credit Card Frau Detection

## Introduction

  It is important that credit card companies are able to recognize frauds on credit card transactions.
  
   On Kaggle, we have access to a dataset which contains transactions made by credit cards in September 2013 by europeans. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
   
   The features V1, V2, ..., V28 are the principal components obtained with PCA and all are numeric and confidentials.

## Problem Statement

Due to the fraudulent credit card transactions problem and your data, how good we can predict them?

To solve this problem, we'll follow a standard data science pipeline plan of attack:

#### 1. Understand the problem and the data
#### 2. Data exploration
#### 3. Feature engineering / feature selection
#### 4. Model evaluation and selection
#### 5. Model optimization
#### 6. Interpretation of results and predictions

### Getting Start:

Doing the necessary imports:


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

__Note__: The last line output the variable without need call __print()__ in the notebook, code can be more clear with this. 

### Reading the dataset:


```python
card_transactions = pd.read_csv('creditcard.csv')
```

## Understand the problem and the data

I will start seeing the shape and columns names of our dataset, to answer my question: How many features and instances do I have?


```python
card_transactions.shape
card_transactions.columns
```




    (284807, 31)






    Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
           'Class'],
          dtype='object')



As mentioned before, the features passed by a PCA algorithm and they are confidentials. The name doesn't help us to understand.

Let's check all feature __types__:


```python
card_transactions.dtypes
```




    Time      float64
    V1        float64
    V2        float64
    V3        float64
    V4        float64
    V5        float64
    V6        float64
    V7        float64
    V8        float64
    V9        float64
    V10       float64
    V11       float64
    V12       float64
    V13       float64
    V14       float64
    V15       float64
    V16       float64
    V17       float64
    V18       float64
    V19       float64
    V20       float64
    V21       float64
    V22       float64
    V23       float64
    V24       float64
    V25       float64
    V26       float64
    V27       float64
    V28       float64
    Amount    float64
    Class       int64
    dtype: object



All of them are numerical and it's coherent. So doesn't need any type of cast.

### Data Distribution

On the Kaggle's challenge description of dataset, they tell this data have a unbalanced distribution. Let's check:


```python
count_classes = pd.value_counts(card_transactions['Class'], sort = True)

# Creating a plot with bar kind:
count_classes.plot(kind = 'bar', rot=0)

# Setting plotting title and axi's legends:
plt.title("Data Class distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc2a3dbb908>






    Text(0.5, 1.0, 'Data Class distribution')






    Text(0.5, 0, 'Class')






    Text(0, 0.5, 'Frequency')




![png](output_15_4.png)


Now I'm sure the data is totally unbalanced.

<font color='red'>__Reminder to the author (me)__: Talk about unbalanced data approachs (collect data, metrics and resampling).</font>

## Data exploration / data cleaning

###  Have any null value in the DataFrame?

I'm going to check if have any value on instances with null values:


```python
card_transactions.isnull().values.any()
```




    False



Well, I don't need to worry about treat null values.

### Analysis Fraud and Valid Transactions 

Determine the number of fraud and valid transactions:


```python
fraud_data = card_transactions[card_transactions['Class'] == 1]
normal_data = card_transactions[card_transactions['Class'] == 0]

print('Fraud shape: ' + str(fraud_data.shape))
print('Valid shape: ' + str(normal_data.shape))
```

    Fraud shape: (492, 31)
    Valid shape: (284315, 31)


#### How many percents each Class represents on this skewed distribution?


```python
print('No Fraud: ', round(len(fraud_data)/len(card_transactions) * 100,2), '% of the dataset.')
print('Fraud: ', round(len(normal_data)/len(card_transactions) * 100,2), '% of the dataset.')
```

    No Fraud:  0.17 % of the dataset.
    Fraud:  99.83 % of the dataset.


#### How different are the amount of money used in different transaction classes?

##### Normal transactions:


```python
fraud_data.Amount.describe()
```




    count     492.000000
    mean      122.211321
    std       256.683288
    min         0.000000
    25%         1.000000
    50%         9.250000
    75%       105.890000
    max      2125.870000
    Name: Amount, dtype: float64



##### Fraud transactions:


```python
normal_data.Amount.describe()
```




    count    284315.000000
    mean         88.291022
    std         250.105092
    min           0.000000
    25%           5.650000
    50%          22.000000
    75%          77.050000
    max       25691.160000
    Name: Amount, dtype: float64



## Feature engineering / feature selection

## Model evaluation and selection

## Model optimization

## Interpretation of results and predictions
