# Machine-Learning-Model
Predicting and Addressing Customer Churn: The case of Model Fitness

Introduction
The loss of clients is a significant issue, and identifying when a customer is no longer engaged can be complex. While some customers might cancel their memberships or fail to renew their contracts, others may leave quietly, making it less obvious that they have disengaged.

In various fields, indicators of customer churn can differ. For this project, we assume that low attendance within a month could signal disengagement and potential loss.

The aim of this project is to design a machine learning model for a gym named Model Fitness to predict churn probabilities based on key features such as location distance, contract length, and attendance to group classes.

These are some of the libraries used in this project:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
```
We had 14 characteristics and 4.000 observations. Churn is our target variable, and the other 13 are characteristics. It's important to add that we ddin't have missing values and all of them are numercial. 

For the model, logistic regression was used as it demonstrated better performance in accuracy, precision, and recall compared to the Random Forest Classifier.

Additionally, a k-means model was applied to classify clients into five clusters: Frequent Spenders, High Risk, Loyal Long Term, Occasional Visitors, and Remote Visitors.

In conclusion, we found that the Loyal Long Term clients, who are more committed (longer contract periods, higher lifetime, frequent class attendance) and have the lowest churn rate, constitute the largest proportion, which is a positive indicator for the gym. However, to decrease churn rates, we could focus on increasing class attendance and extending contract periods through marketing strategies that motivate people to engage in these activities or through promotions for long-term contracts. Additionally, generating attractive strategies for people who live far away from the gym can be beneficial as well.
