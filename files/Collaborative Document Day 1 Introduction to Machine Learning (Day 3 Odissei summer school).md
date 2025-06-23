![](https://i.imgur.com/iywjz8s.png


# Collaborative Document Day 1 Introduction to Machine Learning (Day 3 Odissei summer school)

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.


----------------------------------------------------------------------------

This is the Document for today: https://edu.nl/he7fe

Collaborative Document day 1: https://edu.nl/he7fe

Collaborative Document day 2: https://edu.nl/XXXX


##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.

 If you feel that the code of conduct is breached, please talk to one of the instructors (if the complaint is for one of the participants) or send an email to training@esciencecenter.nl (if the complaint is for one of the instructors).
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

### 🖥 Workshop website
https://esciencecenter-digital-skills.github.io/2025-06-18-ds-sklearn-python-odissei

### 🛠 Setup
https://esciencecenter-digital-skills.github.io/2025-06-18-ds-sklearn-python-odissei#setup

## 👩‍🏫👩‍💻🎓 Instructors

Malte Luken, Flavio Hafner

## 🧑‍🙋 Helpers

Sarah Alidoost, Ou Ku

## 🗓️ Agenda
09:00	Welcome and icebreaker
09:15	Introduction to machine learning
10:00	Break
10:10	Tabular data exploration
11:00	Break
11:10	First model with scikit-learn
12:00	Lunch Break
13:00   Intuitions on linear models
13:10	Fitting a scikit-learn model on numerical data
14:00   Break
14:10	Working with numerical data
15:00	Break
15:10   Model evaluation using cross-validation
15:50	Break
16:00	Guest lecture
17:00	END


## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl.
Please request your certificate within 8 months after the workshop, as we will delete all personal identifyable information after this period.

## 🔧 Exercises
## Exercises

### Exercise: Machine learning concepts 
Given a case study: pricing apartments based on a real estate website. We have thousands of house descriptions with their price. Typically, an example of a house description is the following:

“Great for entertaining: spacious, updated 2 bedroom, 1 bathroom apartment in Lakeview, 97630. The house will be available from May 1st. Close to nightlife with private backyard. Price ~$1,000,000.”

We are interested in predicting house prices from their description. One potential use case for this would be, as a buyer, to find houses that are cheap compared to their market value.

#### What kind of problem is it?

a) a supervised problem :heavy_check_mark: 
b) an unsupervised problem
c) a classification problem :heavy_check_mark: 
d) a regression problem

Select all answers that apply

#### What are the features?

a) the number of rooms might be a feature :heavy_check_mark: 
b) the post code of the house might be a feature :heavy_check_mark: 
c) the price of the house might be a feature

Select all answers that apply

#### What is the target variable?

a) the full text description is the target
b) the price of the house is the target :heavy_check_mark: 
c) only house description with no price mentioned are the target

Select a single answer

#### What is a sample?

a) each house description is a sample :heavy_check_mark: 
b) each house price is a sample
c) each kind of description (as the house size) is a sample

Select a single answer

### 📝 Exercise : Adapting your first model
The goal of this exercise is to fit a similar model as we just did to get familiar with manipulating scikit-learn objects and in particular the `.fit/.predict/.score` API.

Before we used `model = KNeighborsClassifier()`. All scikit-learn models can be created without arguments. This is convenient because it means that you don’t need to understand the full details of a model before starting to use it.

One of the KNeighborsClassifier parameters is n_neighbors. It controls the number of neighbors we are going to use to make a prediction for a new data point.

#### 1. What is the default value of the n_neighbors parameter? 
Hint: Look at the documentation on the scikit-learn website or directly access the description inside your notebook by running the following cell. This will open a pager pointing to the documentation.
```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier?
```

#### 2. Create a KNeighborsClassifier model with n_neighbors=50
a. Fit this model on the train data and target that we used before
b. Use your model to make predictions on the first 10 data points inside the data. Do they match the actual target values?
c. Compute the accuracy on the training data.
d. Now load the test data from "../datasets/adult-census-numeric-test.csv" and compute the accuracy on the test data.

#### 3. (Optional) Find the optimal n_neighbors
What is the optimal number of neighbors to fit a K-neighbors classifier on this dataset?

#### Solutions

Find out default number
```python=
?KNeighborsClassifier # Should be 5
```


Compare accuracy training/test
```python=
predictions_in = model.predict(data) # training data prediction
accuracy_train = model.score(data, target)
accuracy_train
```

```python=
predictions_test = model.predict(data_test) # training data prediction
accuracy_test = model.score(data_test, target_test)
accuracy_test
```

### Comparing Logistic Regression to a Baseline

The goal of this exercise is to compare the performance of our classifier in
the previous notebook (roughly 81% accuracy with `LogisticRegression`) to some
simple baseline classifiers. The simplest baseline classifier is one that
always predicts the same class, irrespective of the input data.

- What would be the score of a model that always predicts `' >50K'`?
- What would be the score of a model that always predicts `' <=50K'`?
- Is 81% or 82% accuracy a good score for this problem?

Use a `DummyClassifier` and do a train-test split to evaluate its accuracy on
the test set. This
[link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
shows a few examples of how to evaluate the generalization performance of
these baseline models.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
```

We first split our dataset to have the target separated from the data used to
train our predictive model.

```python
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)
```

We start by selecting only the numerical columns as seen in the previous
notebook.

```python
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]
```

Split the data and target into a train and test set.

```python
from sklearn.model_selection import train_test_split

# Write your code here.
```

Use a `DummyClassifier` such that the resulting classifier always predict the
class `' >50K'`. What is the accuracy score on the test set? Repeat the
experiment by always predicting the class `' <=50K'`.

Hint: you can set the `strategy` parameter of the `DummyClassifier` to achieve
the desired behavior.

`DummyClassifier` documentation: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

**Solution**

```python=
from sklearn.dummy import DummyClassifier
```

we can have the strategy "most_frequent" to predict the class that appears the most in the training target.

```python=
model = DummyClassifier(strategy="most_frequent", random_state=42)
```

```python=
model.fit(data_train, target_train)
```

```python=
model.predict(data_test)
```

```python=
accuracy = model.score(data_test, target_test)
```

other approach, with `constant`

```python=
model1 = DummyClassifier(strategy="constant", constant=" >50K")
```

#### 2. (optional) Try out other baselines
What other baselines can you think of? How well do they perform?


##### Cross validation 

#### 1. Why do we need two sets: a train set and a test set?

- a) to train the model faster
- b) to validate the model on unseen data :heavy_check_mark: 
- c) to improve the accuracy of the model

Select all answers that apply

#### 2. The generalization performance of a scikit-learn model can be evaluated by:

- a) calling fit to train the model on the training set, predict on the test set to get the predictions, and compute the score by passing the predictions and the true target values to some metric function: :heavy_check_mark: 
- b) calling fit to train the model on the training set and score to compute the score on the test set :heavy_check_mark: 
- c) calling cross_validate by passing the model, the data and the target :heavy_check_mark: 
- d) calling fit_transform on the data and then score to compute the score on the test set

Select all answers that apply

#### 3. When calling `cross_validate(estimator, X, y, cv=5)`, the following happens:

- a) X and y are internally split five times with non-overlapping test sets :heavy_check_mark:
- b) estimator.fit is called 5 times on the full X and y
- c) estimator.fit is called 5 times, each time on a different training set :heavy_check_mark:
- d) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the train sets
- e) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the test sets :heavy_check_mark:

Select all answers that apply

#### 4. (optional) Scaling
We define a 2-dimensional dataset represented graphically as follows:
![](https://i.imgur.com/muvSbI6.png)

Question

If we process the dataset using a StandardScaler with the default parameters, which of the following results do you expect:

![](https://i.imgur.com/t5mTlVG.png)

Select a single answer:

- a) Preprocessing A
- b) Preprocessing B :heavy_check_mark:
- c) Preprocessing C
- d) Preprocessing D


#### 5. (optional) Cross-validation allows us to:

- a) train the model faster
- b) measure the generalization performance of the model :heavy_check_mark:
- c) reach better generalization performance 
- d) estimate the variability of the generalization score :heavy_check_mark:

Select all answers that apply

#### 6. (optional) Explore more options

- look at the documentation of the `cross_validate` function. explore different evaluation metrics than accuracy.
- Using the existing data set, build a full pipeline with scalars, models and cross-validation. Explore different options for cross-validation, following [section 3.1 in the sklearn user guide](https://scikit-learn.org/stable/modules/cross_validation.html).



## 🧠 Collaborative Notes

### Intro to ML

Question: machine learning as inductive vs deductive??

Some terms:
training data can be called data matrix
training data vs testing data - memorize on training data and generalize to test data

**Supervised ML:** from X observations, predict y targets. With a particular goal in mind.
- classification (discrete) and regression (continuous)
- Classification: classify descrete y
- Regression: predict continuous y

**Unsupervied learning:** extract a structure from X that generalizes
can also be used as an early step to extract the most relevant variables, which can then be used in supervised machine learning


**Machine learning algorithms**
- least squares regression
- decision trees (yes/no partitions)
- Support Vector Machines (SVMs)
- Ensemble learning
- Neural Networks (also refered as NN)

**A very simplified relation between the three big words**
AI > ML > Deep Learning

### Manipulate Tablular Data

- creating a notebook and name it as `data_exploration`
- changing the type of the cell to code/markdown (text) and add heading: **Data Exploration**
- loading data with pandas: 
    - path to [the data in setup](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/index.html#download-datasets) 
    - `import pandas as pd`
    -  `adult_census = pd.read_csv("datasets/adult-census.csv")`  you can press `Tab` key on your keyboard to see more option


```python=
import pandas as pd

# Note that the path can vary depends on where you put your datasets
# Essenstially, you need to navigate to "adult-census.csv"
adult_census = pd.read_csv("../datasets/adult-census.csv")
```

```python
adult_census.head()
```

```python=
target_column = "class"
```

```python=
adult_census[target_column].unique()
```

```python=
numerical_columns = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
all_columns = numerical_columns + categorical_columns + [target_column]

adult_census = adult_census[all_columns]
```

visualize as a histogram
```python=
_ = adult_census.hist(figsize = (20, 14))
```

Look into the value distributions
```python=
adult_census[target_column].value_counts()
```

We can also do this to other columns, like "sex"
```python=
adult_census["sex"].value_counts()
```
Inspecting relationships between columns

```python
pd.crosstab(
    index=adult_census["education"], columns=adult_census["education-num"]
)
```

Droping a column:
```python
adult_census = adult_census.drop(
    columns=["education"]
)  # duplicated in categorical column
```

Plotting:
```python
import seaborn as sns

# We plot a subset of the data to keep the plot readable and make the plotting
# faster
n_samples_to_plot = 5000
columns = ["age", "education-num", "hours-per-week"]
_ = sns.pairplot(
    data=adult_census[:n_samples_to_plot],
    vars=columns,
    hue=target_column,
    plot_kws={"alpha": 0.2},
    height=3,
    diag_kind="hist",
    diag_kws={"bins": 30},
)
```

### Fitting a ML model with scikit-learn

```python=
import pandas as pd
import numpy as np
np.set_printoptions(legacy="1.25")
```

```python=
adult_census = pd.read_csv("../datasets/adult-census-numeric.csv")
```

```
adult_census.head()
```

```python=
target_name = "class"
target = adult_census[target_name]
target
```

```python=
data = adult_census.drop(columns = [target_name])
data.head()
```

```python=
data.columns
```

```python=
data.shape
n_rows = data.shape[0]
n_cols = data.shape[1]
```


```python=
print(f"The dataset contains {n_rows} rows and {n_cols} columns.")
```

```python=
from sklearn.neighbors import KNeighborsClassifier
```

Use the KNeighbors classifier with default setup (no arguments)
```python=
model = KNeighborsClassifier()
```

inspect the default setup
```python=
?KNeighborsClassifier
```

fit our first model
```python=
model.fit(data, target)
```

[model_fit](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/_images/api_diagram-predictor.fit.svg)

```python=
target_predicted = model.predict(data)
```

[model_predict](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/_images/api_diagram-predictor.predict.svg)

```python=
target_predicted[:5]
```

```python=
target[:5]
```

```python=
target_predicted[:5] == target[:5]
```

compare the first 5 samples 
```python=
n_correct = (target_predicted[:5] == target[:5]).sum()
print(f"Number of correct predictions {n_correct} / 5")
```

check the percentage of accuracy
```python=
(target == target_predicted).mean()
```

### Train-test split


```python
adult_census_test = pd.read_csv("../datasets/adult-census-numeric-test.csv")
```

```python
target_test = adult_census_test[target_name]
```

```python
data_test = adult_census_test.drop(columns=[target_name])
```

```python=
nrow_test, ncol_test = data_test.shape
print(f"in test data, nrows: {nrow_test}, ncols: {ncol_test}")
```

inspect accuracy
```python=
accuracy = model.score(data_test, target_test)
print(f"The test accuracy is {accuracy}")
```

[model accuracy](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/_images/api_diagram-predictor.score.svg)


### Linear models

- Regression: linear regression
- Classification: logistics regression 


We can create Linear regression with several variables.
And for classification, we can use Logistic regression for multi-class.

**Pros**
- Simple and fast baselines

**Cons**
- perform poorly when n_features << n_samples
- hard to beat when n_features is large

### Working with numerical data

Open a new notebook and do:

```python
import pandas as pd
import numpy as np
```

```python=
adult_census = pd.read_csv("../datasets/adult-census.csv")
```

drop the duplicated column `"education-num"`
```python
adult_census = adult_census.drop(columns="education-num")
adult_census.head()
```

```python
data, target = adult_census.drop(columns="class"), adult_census["class"]
```

look at the target:
```python
target
```

check data types
```python
data.dtypes
```

check unique values:
```python=
data.dtypes.unique()
```

look at the features:
```python=
data.head()
```

Index the dataframe to some columns
```python=
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
```

```python=
data[numerical_columns].head()
```

look at the statistics:
```python=    
data["age"].describe()
```

store the subset of numerical columns in a new dataframe:
```python=
data_numeric = data[numerical_columns]
```

**train-test split**

```python=
from sklearn.model_selection import train_test_split
```

Use the function to split the data:

```python=
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25
) # Here we use 42 as a seed, feel free to use other values!
```

**Train the model:**

```python=
from sklearn.linear_model import LogisticRegression
```

```python=
model = LogisticRegression()
```

```python=
model.fit(data_train, target_train)
```

check the accuracy:
```python=
accuracy = model.score(data_test, target_test)
```

print it:
```python=
print(f"Accuracy of logistic regression: {accuracy:.3f}")
```

### Preprocessing for numerical features

Open a new notebook name it as: `numerical_scaling`

```python
import pandas as pd

# check the correct path
adult_census = pd.read_csv("../datasets/adult-census.csv")
```

```python=
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)
```

select only the numerical columns:
```python=
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]
```

split to train-test
```python=
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42
)
```

printing some statistics about the training data:
```python=
data_train.describe()
```

as can be seen, the ranges in data are different, linear models such as logistic regression generally benefit from scaling the features.

```python=
from sklearn.preprocessing import StandardScaler
```

This scaler shifts and scales each feature individually so that they all have a 0-mean and a unit standard deviation.

```python=
scaler = StandardScaler()
```

We use `fit` method to learn the scaling from the training data: standard deviation is calculated from training data and used for test data. Otherwise, the scaling parameters might be calculated from test data and overfitting happens.

```python=
scaler.fit(data_train)
```

inspect the computed means:
```
scaler.mean_
```

inspect the computed std_dev:
```python=
scaler.scale_
```

perform data transformation by:
```python=
data_train_scaled = scaler.transform(data_train)
data_train_scaled
```
The result here is a numpy array.


shortcut for fit and transform at the same time:
```python=
data_train_scaled = scaler.fit_transform(data_train)
```

This will return a pandas dataframe:
```
scaler = StandardScaler().set_output(transform="pandas")
data_train_scaled = scaler.fit_transform(data_train)
```



inspect the results:
```python=
data_train_scaled.describe()
```

plot:

```python=
import matplotlib.pyplot as plt
import seaborn as sns

# number of points to visualize to have a clearer plot
num_points_to_plot = 300

sns.jointplot(
    data=data_train[:num_points_to_plot],
    x="age",
    y="hours-per-week",
    marginal_kws=dict(bins=15),
)
plt.suptitle(
    "Jointplot of 'age' vs 'hours-per-week' \nbefore StandardScaler", y=1.1
)

sns.jointplot(
    data=data_train_scaled[:num_points_to_plot],
    x="age",
    y="hours-per-week",
    marginal_kws=dict(bins=15),
)
_ = plt.suptitle(
    "Jointplot of 'age' vs 'hours-per-week' \nafter StandardScaler", y=1.1
)
```

as can be seen, scaling does not change distribution and correlation of the data. 

**Pipeline**

We can combine sequential operations with a scikit-learn `Pipeline`:

```python=
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
```

```python=
model = make_pipeline(StandardScaler(), LogisticRegression())
model
```
Here we can see different steps of our pipeline.

check the name of each steps:
```python=
model.named_steps
```

fit and predict:
```python=
model.fit(data_train, target_train)
```

```python=
predicted_target = model.predict(data_test)
```
```python
predicted_target[:5]
```

and use `score` method for accuracy:

```python=
accuracy = model.score(data_test, target_test)
```


### Cross-validation

```python=
import pandas as pd
import numpy as np

# check the path to the data
adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)
```

```python=
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]
```

import some libraries:
```python=
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
```

create the pipeline:
```python=
model = make_pipeline(StandardScaler(), LogisticRegression())
```

In the figure below, you can see how cross validation works:
![image](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/_images/cross_validation_diagram.png)

import the librray for cross validation
```python=
from sklearn.model_selection import cross_validate
```

```python=
cv_result = cross_validate(model, data_numeric, target, cv=5)
```

inspect the results
```python=
cv_result
```

let’s extract the scores computed:
```python=
scores = cv_result["test_score"]
```

and mean and std_dev:
```python=
scores.mean()
scores.std()

```


## 📚 Resources

## Debug env setup

Incase in Windows, CommandLine you run into something like: "execution of scripts is disabled"

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
