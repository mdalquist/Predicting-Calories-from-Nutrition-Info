# Predicting Number of Calories in a Recipe Based on Nutrition Info


## Framing the Problem

For this project, I will be predicting the number of calories in a recipe from the nutrition info. The nutrition info is given in the 'recipes' dataframe in a column labeled 'nutrition'. The nutrition column contains lists with the following information, all measured in percentage of daily value (except for calories, it is listed as its true value): calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates. I will use linear regression, with calories as the response variable and the other nutrition info being used as numerical features. Total Fat, Sugar, Sodium, Protein, Saturated Fat, and Carbohydrates are all known at the time of prediction, so all are valid to use to predict Calores. Root Mean Squared Error (RMSE) will be used to evaluate model performance, as it is a numerical metric that can be used to tell how far off the predictions are. This is more suitable for this question than R^2, which determines the strength of the fit, because I am more concerned with the value of the prediction than the overall accuracy of the trendline.

Datasets, cleaning, and Exploratory Data Analysis are performed [here](https://mdalquist.github.io/Practice-on-Food-Dataset/). The data cleaning steps are repeated and modified for the new problem on this site.

All of the data needed from the original datasets needed for this project is contained in the 'nutrition' column, seen below.

|    | nutrition|
|---:|:---------------------------------------------|
|  0 | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     |
|  1 | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |
|  2 | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |
|  3 | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |
|  4 | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |


## Baseline Model

