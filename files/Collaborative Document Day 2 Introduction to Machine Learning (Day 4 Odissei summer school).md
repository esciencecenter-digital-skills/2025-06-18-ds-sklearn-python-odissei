![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 2 Introduction to Machine Learning (Day 4 Odissei summer school)

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://edu.nl/ngjrx

Collaborative Document day 1: https://edu.nl/he7fe

Collaborative Document day 2: https://edu.nl/ngjrx


##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.

For more details, see [here](https://docs.carpentries.org/policies/coc/).

Want to report a Code of Conduct incident and you prefer to do it anonymously? You can do it [here](https://goo.gl/forms/KoUfO53Za3apOuOK2).

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a yellow post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## Links

### 🖥 Workshop website
https://esciencecenter-digital-skills.github.io/2025-06-18-ds-sklearn-python-odissei

### 🛠 Setup
https://esciencecenter-digital-skills.github.io/2025-06-18-ds-sklearn-python-odissei#setup

## 👩‍🏫👩‍💻🎓 Instructors

Malte Luken, Flavio Hafner

## 🧑‍🙋 Helpers

Fakhereh (Sarah) Alidoost, Ou Ku

## 🗓️ Agenda
09:00	Welcome and recap
09:15	Encoding categorical variables
10:00	Break
10:10	Using numerical and categorical variables together
11:10	Break
11:20   Overfitting and underfitting
11:45   Intuitions on decision trees
12:10	Lunch Break
13:10   Validation and learning curves
~~13:30	Advanced topics~~
14:10   Break
14:20	Try out learned skills
14:50   Break
15:00	Guest lecture
16:10	Concluding remarks/best practices
16:45   Wrap up
17:00	END


## 🔧 Exercises

### Machine learning 30 seconds

Explain as many of the 5 concepts in 30 seconds without (partially) naming the concepts.

#### Round 1

- Predictive model
- Test data v
- Classification v
- Ordinal encoding
- Cross-validation


#### Round 2

- Data matrix
- Regression
- Standardization
- One-hot encoding
- K-fold

### Handling categorical data 

#### Ordinal encoding (5 minutes in pairs, then discussion): 

Q1: Is ordinal encoding appropriate for marital status? :heavy_multiplication_x:  For which (other) categories in the adult census would it be appropriate? Why?
> Only education (in fact, the encoder was already present in the data set as education-num), as this is the only one that can be expressed as an incremental feature

Q2: Can you think of another example of categorical data that is ordinal?
Q3: What problem arises if we use ordinal encoding on a sizing chart with options: XS, S, M, L, XL, XXL? (HINT: explore `ordinal_encoder.categories_`)

> Would not be in correct order (it's alphabetized). And they may not be comparible between different sources.

Q4: How could you solve this problem? (Look in documentation of OrdinalEncoder)

> top of documention will tell you to use `categories` argument with a list in the correct order
```
ordered_size_list = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
encoder_with_order = OrdinalEncoder(categories=ordered_size_list)
```

Q5: Can you think of an ordinally encoded variable that would not have this issue?

> US grading scheme is alphabetical to begin with (A,B,C,D,F)


#### Exercise: The impact of using integer encoding for with logistic regression (groups of 2, 15min)


Goal: understand the impact of arbitrary integer encoding for categorical variables with linear classification such as logistic regression.

We keep using the `adult_census` data set already loaded in the code before. Recall that `target` contains the variable we want to predict and `data` contains the features.

If you need to re-load the data, you can do it as follows:

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```


**Q0 Select columns containing strings**
Use `sklearn.compose.make_column_selector` to automatically select columns containing strings that correspond to categorical features in our dataset.

**Q1 Build a scikit-learn pipeline composed of an `OrdinalEncoder` and a `LogisticRegression` classifier**

You'll need the following, already loaded modules:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
```

Because OrdinalEncoder can raise errors if it sees an unknown category at prediction time, you can set the handle_unknown="use_encoded_value" and unknown_value parameters. You can refer to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) for more details regarding these parameters.

**Answer Q1**

```python=
model_ordinal = make_pipeline(
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    LogisticRegression(max_iter=500)
)
```


**Q2 Evaluate the model with cross-validation.**

You'll need the following, already loaded modules:

```python
from sklearn.model_selection import cross_validate

```

**Answer Q2**

```python=
cv_results_ordinal = cross_validate(model_ordinal, data_categorical, target)
cv_results_ordinal
```

> As we see in the results, the accuracy decreases to ~0.75. This shows using the wrong encoding can lower the quality of training.

**Q3 Repeat the previous steps using an `OneHotEncoder` instead of an `OrdinalEncoder`**

You'll need the following, already loaded modules:

```python
from sklearn.preprocessing import OneHotEncoder

```

### Exercise: overfitting and underfitting

#### 1: A model that is underfitting:

- a) is too complex and thus highly flexible
- b) is too constrained and thus limited by its expressivity :heavy_check_mark: 
- c) often makes prediction errors, even on training samples :heavy_check_mark:
- d) focuses too much on noisy details of the training set

Select all answers that apply

#### 2: A model that is overfitting:

- a) is too complex and thus highly flexible :heavy_check_mark:
- b) is too constrained and thus limited by its expressivity
- c) often makes prediction errors, even on training samples
- d) focuses too much on noisy details of the training set :heavy_check_mark:

Select all answers that apply

### Exercise: Validation and learning curves

#### Train and test SVM classifier 

The aim of this exercise is to:
* train and test a support vector machine classifier through cross-validation;
* study the effect of the parameter gamma (one of the parameters controlling under/over-fitting in SVM) using a validation curve;
* determine the usefulness of adding new samples in the dataset when building a classifier using a learning curve. 

We will use blood transfusion dataset located in `../datasets/blood_transfusion.csv`. First take a data exploration to get familiar with the data.

You can then start off by creating a predictive pipeline made of:

* a [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) with default parameter;
* a [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

Script below will help you get started:

```python=
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC())
```

You can vary gamma between 10e-3 and 10e2 by generating samples on a logarithmic scale with the help of

```python=
import numpy as np
gammas = np.logspace(-3, 2, num=30)
param_name = "svc__gamma"
```

To manipulate training size you could use:

```python=
train_sizes = np.linspace(0.1, 1, num=10)
```
Evaluate the effect of the parameter gamma by using `sklearn.model_selection.ValidationCurveDisplay`. Compute the learning curve (using `sklearn.model_selection.LearningCurveDisplay`):
```python
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.model_selection import LearningCurveDisplay

```
#### Solution

**Preparing the data**

```python=
import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]
```

```python=
# solution
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC())
```

```python=
import numpy as np
gammas = np.logspace(-3, 2, num=30)
param_name = "svc__gamma"
```

```python=
train_sizes = np.linspace(0.1, 1, num=10)
```

```python=
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
cv_results["test_score"].mean()
```


**The validation curve**

```python=
import numpy as np

from sklearn.model_selection import ValidationCurveDisplay

disp = ValidationCurveDisplay.from_estimator(
    model,
    data,
    target,
    param_name=param_name,
    param_range=gammas,
    cv=5,
    scoring="accuracy", 
    score_name="Accuracy",
    std_display_style="errorbar",
    n_jobs=2,
)
```

**The learning curve**
```python=
from sklearn.model_selection import LearningCurveDisplay

LearningCurveDisplay.from_estimator(
    model,
    data,
    target,
    train_sizes=train_sizes,
    cv=5,
    score_type="both",
    scoring="accuracy",  
    score_name="Accuracy",
    std_display_style="errorbar",
    n_jobs=2,
)
```

### 1st option_Exercise: Try out learned skills on same dataset: `adult_census`

#### Scores

### 2nd option_Exercise: Try out learned skills on Ames Housing dataset
In this exercise we use the Ames Housing dataset.

We use this dataset in a regression setting to predict the sale prices of houses based on house features. That is, the goal is to predict the target `SalePrice` from numeric and/or categorical features in the dataset.

Remember to explore the dataset before building models. Then, start simple and step-by-step expand your approach to create better and better models.

You can load the data as follows:
```python
house_prices = pd.read_csv("../datasets/ames_housing_no_missing.csv")
```

#### In case exercise is difficult

To help you get started, you can take a look at the following link: https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_ames_housing.html

#### Scores

- Malte: 0.869 +/- 0.02


## 🧠 Collaborative Notes

[link to the dataset](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/#download-datasets)

### Categorial Variables

```python=
import pandas as pd

adult_census = pd.read_csv("datasets/adult-census.csv")

target_name = "class"

target = adult_census[target_name]

data = adult_census.drop(columns=[target_name])
```

```python=
data.head()
```

```python=
data["native-country"].value_counts()
```

```python=
data.dtypes
```

```python=
from sklearn.compose import make_column_selector as selector
```

```python=
categorical_columns_selector = selector(dtype_include=object)

categorical_columns = categorical_columns_selector(data)
categorical_columns
```

```python=
data_categorical = data[categorical_columns]

data_categorical.head()
```

**Encoding ordinal categories**

```python=
from sklearn.preprocessing import OrdinalEncoder
```

```python=
education_column = data_categorical[["education"]]
```

```python=

encoder = OrdinalEncoder().set_output(transform="pandas") # Use "set_output" to make sure encoder returns a pandas.DataFrame. This allows us using pandas functions later
# You can give a list into OrdinalEncoder to sort

education_encoded = encoder.fit_transform(education_column)
```

```python=
education_encoded
```

check the catagories
```python=
encoder.categories_
```

If you would like to specify the order manually
```python=
encoder_manual = OrdinalEncoder(categories=[[' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th',
        ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate',
        ' HS-grad', ' Masters', ' Preschool', ' Prof-school',
        ' Some-college']]).set_output(transform="pandas")

encoder_manual.fit_transform(education_column)

encoder_manual.categories_
```

Create a map between numerical values and categories:
```python=
mapping = {index: cat for index, cat in enumerate(encoder.categories_[0])}
mapping
```

**OneHotEncoder**
For a given feature, it creates as many new columns as there are possible categories. For a given sample, the value of the column corresponding to the category is set to 1 while all the columns of the other categories are set to 0.

```python=
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
```

```python=
education_encoded = encoder.fit_transform(education_column)
education_encoded
```

```python=
education_encoded.shape
```

```python=
data["native-country"].value_counts()
```

We can ignore all unknown categories during `transform` as:
```python=
encoder = OneHotEncoder(handle_unknown="ignore")
```

`unknown_value=-1` would set all values encountered during transform to -1 which are not part of the data encountered during the `fit` call.

```python=
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```
Create a machine learning pipeline: 

```python=
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


model = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)) # By using 500 give it more chances to find the optimal solutions 
```

```python=
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data_categorical, target)
cv_results
```

### Using numerial and categorical variables together

Data type `object` corresponds to categorical columns (strings). We make use of `make_column_selector` helper to select the corresponding columns:

```python=
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)
```

```python=
from sklearn.preprocessing import StandardScaler
```

```python=
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()
```

```python=
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    (numerical_preprocessor, numerical_columns)
)
```

![image](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/_images/api_diagram-columntransformer.svg)

```python=
model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
```

```python=
cv_results = cross_validate(model, data, target)
cv_results
```

let's have a look at the `mean`:

```python=
cv_results["test_score"].mean()
```

### Overfitting and underfitting

Understand when and why a model does or does not generalize well on unseen data.

**Varying model complexity**:
- Data generated by a random process
- This process is unknown
- We can only access the observations
- Fit polynomials of various degrees

Take home messages:
**overfit**: model too complex 
**Underfit**: model too simple

How to find the right trade-off?

### Decision Trees

A decision tree is a set of rules, combined in a hierarchical manner. It can be used for both classification and regression problems.

Take home messages:
- Decision Tree of simple decision rules (one threshold t a time)
- No scaling required of numerical features
- They are building blocks for more complicated models, like Random Forest and Gradient Boosting Decision Trees

### Validation and learning curves

- Varying complexity: validation curves
- Varying the sample size: learning curves

**Comparing train and test errors**:
To understand the overfitting / underfitting trade-off.

Bayes error rate: when adding more data does not provide improvements --> it is useful to try more complex models.

Different model families come with different forms of complexity and bias (which we call inductive bias).


### Concluding remark

**The machine learning pipeline**:
- learned on a train set and then applied to new data, a “test set”
- Scikit-learn models are built from a data matrix
- Transformations of the data are often necessary

**Adapting model complexity to the data**:
- Models seek to minimize the error on the test set
- Models come with multiple hyper-parameters

**Specific models**


**Validation and evaluation matter**:
- As you narrow down on a solution, spend increasingly more effort on validating it

**Machine learning is a small part of the problem most of the times**

**Technical craft is not all**:
Once you know how to run the software, the biggest challenges are understanding the data, its shortcomings, and what can and cannot be concluded from an analysis.

**How the predictions are used**
- Errors mean different things

**Biases in the data**

**Prediction models versus causal models**


## 📚 Resources

- Lesson material with slides: https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/trees/slides.html
- sklearn documentation on Support Vector Machine
- [Going further with machine learning](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/concluding_remarks.html#going-further-with-machine-learning)

## Post-workshop survey

[Here is the link to the survey.](https://www.surveymonkey.com/r/3K97BST)

