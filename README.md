## Hyper Parameter Tuning - GridSearchCV Usecase 

This project uses 'digits dataset' to identify the best function/classifier with the unique combination of parameters using GridSearchCV function. 

It has compared the results of Support Vactor Machine, Random Forest Classifier, Naives Bayes (Gaussian and Multinomial), Logistic Regression, and Decision Tree. 

### Project Workflow
1. Loaded the Digits dataset using Scikit-learn.
2. Created dictionary for the testing models along with their parameters:

 - **Suport Vector Machine** (kernel= [rbf, linear], C= [1, 10, 20])
 - **Random Forest Classifier** (n_estimator= [1, 5, 10])
 - **Logistic Regression** (solver= liblinear, max_iter= 1000)
 - **Gaussian Bayes** (var_smoothing= 1e-9, 1e-8, 1e-7)
 - **Multinomial Bayes** (alpha= [0.1, 0.5, 1.0], fit_prior= True, False)
 - **Decision Tree** (max_depth= [3, 5, 10, None],
min_samples_split= [2, 5, 10])

3. Used a 'for' loop for GridSearchCV functions' result
4. Appended the 'model name', 'accuracy score', and 'best parameters' columns to get the desired results in a dataframe. 

### Libraries Used
- Pandas: For data manipulation. 
- Scikit-learn: For machine learning algorithms and cross-validation.

```bash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
```
### Installation
```bash
pip install pandas scikit-learn
```
### Result
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>best_score</th>
      <th>best_params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>svm</td>
      <td>0.947697</td>
      <td>{'C': 1, 'kernel': 'linear'}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>random_forest</td>
      <td>0.904856</td>
      <td>{'n_estimators': 10}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>logistic_regression</td>
      <td>0.922114</td>
      <td>{'C': 1}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gaussian_bayes</td>
      <td>0.832518</td>
      <td>{'var_smoothing': 1e-07}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>multinomial_bayes</td>
      <td>0.870907</td>
      <td>{'alpha': 0.1, 'fit_prior': True}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>decision_tree</td>
      <td>0.787465</td>
      <td>{'max_depth': None, 'min_samples_split': 2}</td>
    </tr>
  </tbody>
</table>
</div>

Thus for Digits dataset 'support vector machine' with linear kernel, and C= 1 proves to be more efficient, giving the highest score. 
