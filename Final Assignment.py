##  This is my final assignment for the Analyzing Data With Python course from IBM. In this assignment, I analyzed and predicted 
##  alcohol consumption by country using attributes or features such as beer servings and wine servings. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

## 1. Load the csv:
df= pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/edx/project/drinks.csv')

## 2. We use the method head() to display the first 5 columns of the dataframe:
df.head(5)

## 3. Display the data types of each column using the attribute dtype:
df.dtypes

## 4. Use the method groupby to get the number of wine servings per continent:
df_test1 = df[["wine_servings", "continent"]]
df_grp1 = df_test1.groupby(["continent"], as_index = False).mean()
df_grp1

## 5. Perform a statistical summary and analysis of beer servings for each continent:
df_test2 = df[["beer_servings", "continent"]]
df_grp2 = df_test2.groupby(["continent"])
df_grp2.describe()

## 6. Use the function boxplot in the seaborn library to produce a plot that can be used to show the number of beer servings on 
##    each continent:
bplot = sns.boxplot(y=df["beer_servings"], x = df["continent"])

## 7. Use the function regplot in the seaborn library to determine if the number of wine servings is negatively or positively 
##    correlated with the number of beer servings:
sns.regplot(x = "beer_servings", y = "wine_servings", data = df)
plt.ylim(0,)
plt.show()

pears_coef, p_val = stats.pearsonr(df["beer_servings"], df["wine_servings"]) ##I also used Pearson correlation to confirm the correlation

print("The pearson coefficient of", pears_coef, "implies there is a weak positive correlation and a p-value of", p_val, "gives strong certainty.")
print("Therefore the number of wine servings appears to be positively correlated with the number of beer servings, as seen in the graph.")

##6. Fit a linear regression model to predict the 'total_litres_of_pure_alcohol' using the number of 'wine_servings' then 
##  calculate  R^2:
lm = LinearRegression()

x = df[["wine_servings"]]
y = df[["total_litres_of_pure_alcohol"]]

lm.fit(x, y)
y_hat = lm.predict(x)

print("The coefficient is:", lm.coef_, "and the intercept is:", lm.intercept_)
print("The value of R^2 is", lm.score(x, y))

##7. Use the list of features to predict the 'total_litres_of_pure_alcohol', split the data into training and testing and determine the
##  R^2  on the test data:
x_data = df[["beer_servings", "spirit_servings", "wine_servings"]]
y_data = df["total_litres_of_pure_alcohol"]

lm.fit(x_data, y_data)
lm.predict(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 0)

scores = cross_val_score(lm, x_test, y_test, cv = 3)

print("The R^2 value on the test data is:", scores.mean())

##7. Create a pipeline object that scales the data, performs a polynomial transform and fits a linear regression model. Fit the 
##   object using the training data in the question above, then calculate the R^2 using the test data:
input = [("scale", StandardScaler()), ("polynomial", PolynomialFeatures(include_bias=False)), ("mode", LinearRegression())]
pipe = Pipeline(input)

pipe.fit(x_train, y_train)
lm.score(x_test, y_test)

##8. Create and fit a Ridge regression object using the training data, setting the regularization parameter to 0.1 and calculate the
##   R^2 using the test data:
RidgeModel_q9 = Ridge(alpha = 0.1)

RidgeModel_q9.fit(x_train, y_train)
RidgeModel_q9.score(x_test, y_test)
