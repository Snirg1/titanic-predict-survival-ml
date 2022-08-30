# # The main function implements an end-to-end classification pipeline including mainly a data EDA and prediction parts.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.preprocessing import OneHotEncoder
plt.ion()


## Load train.csv
def load_train_data():
    df_train = pd.read_csv('train.csv')
    return df_train


## Display some data - display the top 10 rows
def disp_some_data(df_train):
    print(df_train.head(10))
    return


## In order to know what to do with which columns, we must know what types are there, and how many different values are there for each.
def display_column_data(df_train, max_vals=10):
    info = df_train.info()
    print(info)

    num_uq_vals_sr = df_train.nunique()
    print(num_uq_vals_sr)

    mask = num_uq_vals_sr < max_vals
    columns_to_print = num_uq_vals_sr[mask].index

    for col in columns_to_print:
        print('{:s}: '.format(col), dict(df_train[col].value_counts()))
    return


## Now that we know which columns are there, we can drop some of them - the ones that do not carry predictive power.
## In addition we will drop columns that we do not know how to handle such as free text.
## Drop the columns: PassengerId, Name, Ticket, Cabin
def drop_non_inform_columns(df_train):
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_lean = df_train.drop(columns=columns_to_drop)
    return df_lean


# Now that we know the basics about our dataset, we can start cleaning & transforming it towards enabling prediction of survival
# In which columns are there missing values?
def where_are_the_nans(df_lean):
    mask = df_lean.isna().any()
    cols_with_nans_names = df_lean.columns[mask]
    num_of_no_existing_values = df_lean[cols_with_nans_names].count()
    num_of_missing_values = len(df_lean.index) - num_of_no_existing_values
    cols_with_nans = dict(zip(cols_with_nans_names, num_of_missing_values))
    print(cols_with_nans)

    return cols_with_nans


# We see that the columns 'Age' and 'Embarked' have missing values. We need to fill them.
# Let's fill 'Age' with the average and 'Embarked' with the most common
def fill_titanic_nas(df_lean):
    average_age = df_lean['Age'].mean()
    max_embarked_value = df_lean['Embarked'].value_counts().idxmax()
    df_lean['Age'].fillna(value=average_age, inplace=True)
    df_lean['Embarked'].fillna(value=max_embarked_value, inplace=True)
    df_filled = df_lean
    return df_filled


# ## Now that we filled up all the missing values, we want to convert the non-numerical (categorical) variables
# ## to some numeric representation - so we can apply numerical schemes to it.
# ## We'll encode "Embarked" and "Pclass", using the "one-hot" method
def encode_one_hot(df_filled):
    encoder = OneHotEncoder(handle_unknown='ignore')

    encoded_embarked = pd.DataFrame(encoder.fit_transform(df_filled[['Embarked']]).toarray())
    res = df_filled.join(encoded_embarked)
    res.rename(columns={0: "Emb_C", 1: "Emb_Q", 2: "Emb_S"}, inplace=True)

    encoded_pclass = pd.DataFrame(encoder.fit_transform(df_filled[['Pclass']]).toarray())
    res = res.join(encoded_pclass)
    res.rename(columns={0: "Cls_1", 1: "Cls_2", 2: "Cls_3"}, inplace=True)

    df_dummies = pd.get_dummies(res["Sex"])
    res = pd.concat((df_dummies, res), axis=1)
    res = res.drop(["male"], axis=1)
    res.rename(columns={'female': "Bin_sex"}, inplace=True)

    df_one_hot = res
    return df_one_hot


## There are 2 variables (columns) that reflect co-travelling family of each passenger.
## SibSp - the number of sibling - brothers and sisters.
## Parch - the total number of parents plus children for each passneger.
## We want to reflect the whole family size of each passenger - the sum of SibSp and Parch
## It will be useful later

def make_family(df_one_hot):
    family_total_size = df_one_hot['SibSp'] + df_one_hot['Parch']
    df_one_hot['Family'] = family_total_size
    return df_one_hot


# ## Feature Transformation
# ## In many cases, it is the *multiplicative* change in some numeric variable that affects the outcome, not the *additive* one.
# ## For example, we expect that the change of survival probability of 16 year olds relative to 12 year olds is *much greater*
# ## than the change of survival probability of 48 year olds relative to 44 year olds.
# ## To capture that notion, we take the log of the variables of interest. To guard against taking log of 0, we add 1 prior to that.
# ## All in all, it produces not bad results.
# ## In short: X -> log(1+X)
# ## There is a numpy function exactly for that: np.log1p
# ## This will be useful later
def add_log1p(df_one_hot):
    for col in ['Age', 'SibSp', 'Parch', 'Fare', 'Family']:
        df_one_hot['log1p_' + col] = np.log1p(df_one_hot[col])
    return df_one_hot


# Basic exploration of survival.
# This section deals with correlations of the "Survived" column to various other data about the passengers.
# Also, in this section, we can still use the df_filled DataFrame, without "one-hot" encoding. It is up to you.
## Survival vs gender
def survival_vs_gender(df):
    df_male = df[df['Bin_sex'] == 0]
    df_female = df[df['Bin_sex'] == 1]

    mean_survived_male = df_male["Survived"].mean()
    mean_survived_female = df_female["Survived"].mean()
    survived_by_gender = {"male": mean_survived_male, "female": mean_survived_female}
    print(survived_by_gender)
    return survived_by_gender


##  The same for survival by class. You can use the "one-hot" encoding, or the original "Pclass" column - whatever more convenient to you.
def survival_vs_class(df):
    df1 = df[df['Pclass'] == 1]
    df2 = df[df['Pclass'] == 2]
    df3 = df[df['Pclass'] == 3]
    mean1 = df1["Survived"].mean()
    mean2 = df2["Survived"].mean()
    mean3 = df3["Survived"].mean()
    survived_by_class = {"Cls_1": mean1, "Cls_2": mean2, "Cls_3": mean3}
    print(survived_by_class)

    return survived_by_class


## The same, for survival by the three family size metrics. Return a dict of dicts / series
def survival_vs_family(df):
    survived_by_family = {}
    highest = 0
    highest_chance_metric = ''

    for metric in ["SibSp", "Parch", "Family"]:
        metric_dict_keys = df[metric].unique()
        metric_dict_keys.sort()
        survived_by_metric = {}

        for key in metric_dict_keys:
            df_metric_and_survived = df[df[metric] == key]
            metric_survived_mean = df_metric_and_survived["Survived"].mean()
            survived_by_metric[key] = metric_survived_mean

        print("Family metric: ", metric)
        print("Survival stats:")
        print(survived_by_metric)

        survived_by_family[metric] = survived_by_metric

    chances_by_metric = list(survived_by_metric.values())
    highest_chance_by_metric = metric_dict_keys[chances_by_metric.index(max(chances_by_metric))]
    if highest_chance_by_metric > highest:
        highest = highest_chance_by_metric
        highest_chance_metric = metric

    print("To ensure the highest chance of survival, the metric ", highest_chance_metric, 'must have the value ',
          highest)
    return survived_by_family


## Visualizing the distribution of age and its impact on survival
def survival_vs_age(df):
    sur_age_df = df[['Age', 'Survived']]

    survived_df = sur_age_df[sur_age_df['Survived'] == 1]
    plt.close('Age, Survived')
    plt.figure('Age, Survived')
    survived_df['Age'].hist(bins="auto")

    not_survived_df = sur_age_df[sur_age_df['Survived'] == 0]
    plt.close('Age, Not Survived')
    plt.figure('Age, Not Survived')
    not_survived_df['Age'].hist(bins="auto")
    
    plt.show(block=True)
    return


# Correlation of survival to the numerical variables
# ['Age', 'SibSp', 'Parch', 'Fare', 'Family']
# ['log1p_Age', 'log1p_SibSp', 'log1p_Parch', 'log1p_Fare', 'log1p_Family']
def survival_correlations(df):
    corr = df.corr()
    # corr is a DataFrame that represents the correlatio matrix
    print(corr)

    df = df[
        ['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Family', 'log1p_Age', 'log1p_SibSp', 'log1p_Parch', 'log1p_Fare',
         'log1p_Family']]
    corr = df.corr()["Survived"][:]
    corr = corr.drop(labels=['Survived'])

    print(corr)
    abs_corr = corr.abs()
    most_important_abs = abs_corr.nlargest(n=5)
    most_important_keys = most_important_abs.keys()
    res = {}
    for k in most_important_keys:
        res[k] = corr[k]

    important_corrs = res
    print(important_corrs)

    return important_corrs


# Predicting survival!!!
# We're finally ready to build a model and predict survival!
# In this section, df_one_hot include all the transformations and additions we've done to the data, including, of course, the one-hot
# encoding of class and port of boarding (Embarked), and the binary "Sex" column, and also with the addition of the log1p scaled variables.
# But including too much features, not to metntion redundant ones (think log1p_Age and Age), can deteriorate the performance.
# Based on the correlations of the numeric data to survival, and the impact of the categorical data, we will pick the best features
# that will yield the best testing results.

##  split data into train and test sets
def split_data(df_one_hot):
    from sklearn.model_selection import train_test_split
    df_one_hot = df_one_hot.drop(
        labels=['log1p_Age', 'log1p_SibSp', 'log1p_Parch', 'Fare', 'Family', 'Sex', 'Embarked', 'Pclass'], axis=1)

    Y = df_one_hot['Survived']
    X = df_one_hot.drop(['Survived'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test


##  Training and testing!
def train_logistic_regression(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    para_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 50],  # internal regularization parameter of LogisticRegression
                 'solver': ['sag', 'saga']}

    Logit1 = GridSearchCV(LogisticRegression(penalty='l2', random_state=1), para_grid, cv=5)

    Logit1.fit(X_train, y_train)

    y_test_logistic = Logit1.predict(X_test)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score

    conf_matrix = confusion_matrix(y_test, y_test_logistic)
    accuracy = accuracy_score(y_test, y_test_logistic)
    f1_score = f1_score(y_test, y_test_logistic)

    print('acc: ', accuracy, 'f1: ', f1_score)
    print('confusion matrix:\n', conf_matrix)

    return accuracy, f1_score, conf_matrix


def warn(*args, **kargs):
    pass


if __name__ == '__main__':
    
    import warnings

    warnings.warn = warn
    df_train = load_train_data()
    disp_some_data(df_train)
    display_column_data(df_train, max_vals=10)
    df_lean = drop_non_inform_columns(df_train)

    cols_with_nans = where_are_the_nans(df_lean)
    df_filled = fill_titanic_nas(df_lean)
    df_one_hot = encode_one_hot(df_filled)
    df_one_hot = make_family(df_one_hot)
    df_one_hot = add_log1p(df_one_hot)

    survived_by_gender = survival_vs_gender(df_one_hot)
    survived_by_class = survival_vs_class(df_one_hot)
    survived_by_family = survival_vs_family(df_one_hot)
    survival_vs_age(df_one_hot)
    important_corrs = survival_correlations(df_one_hot)

    X_train, X_test, y_train, y_test = split_data(df_one_hot)
    train_logistic_regression(X_train, X_test, y_train, y_test)
