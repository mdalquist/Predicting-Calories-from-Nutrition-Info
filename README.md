# Predicting Number of Calories in a Recipe Based on Nutrition Info


## Framing the Problem

For this project, I will be predicting the number of calories in a recipe from the nutrition info. The nutrition info is given in the 'recipes' dataframe in a column labeled 'nutrition'. The nutrition column contains lists with the following information, all measured in percentage of daily value (except for calories, it is listed as its true value): calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates. I will use linear regression, with calories as the response variable and the other nutrition info being used as numerical features. Total Fat, Sugar, Sodium, Protein, Saturated Fat, and Carbohydrates are all known at the time of prediction, so all are valid to use to predict Calores. Root Mean Squared Error (RMSE) will be used to evaluate model performance, as it is a numerical metric that can be used to tell how far off the predictions are. This is more suitable for this question than R^2, which determines the strength of the fit, because I am more concerned with the value of the prediction than the overall accuracy of the trendline.

Datasets, cleaning, and Exploratory Data Analysis are performed [here](https://mdalquist.github.io/Practice-on-Food-Dataset/). The data cleaning steps are repeated and modified for the new problem on this site.

The original data comes from [food.com](food.com), where contributors can add their recipes to the site, including the ingredients, steps, prep time, and nutrition facts. Users of the site can also leave reviews. All of the data needed from the original datasets needed for this project is contained in the 'nutrition' column of the recipes dataframe, seen below. Each entry is a string looking like a list, with the nutrition facts in the order: calores, total fat, sugar, sodium, protein, saturated fat, carbohydrates.

|    | nutrition|
|---:|:---------------------------------------------|
|  0 | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     |
|  1 | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |
|  2 | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |
|  3 | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |
|  4 | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |

Below is the final cleaned dataframe used to build the regression models. It was obtained by isolating the splitting the strings from the nutrition column and adding a new column for each nutrient. Also, each column is converted from percentage of daily value to grams. The conversions come from the [FDA](https://netrition.com/pages/reference-values-for-nutrition-labeling) daily recommendations. These values are sometimes changed by the FDA, so using the wrong conversion can cause error. Using an example from [food.com](https://www.food.com/recipe/chickpea-and-fresh-tomato-toss-51631), I believe the website uses old recommendations of 65 grams of total fat and 25 grams of added sugar daily, which differ from the current recommendations of 78g and 50g, respectively. The other recommended daily values are: sodium, 2.3g; protein, 50g; saturated fat, 20g; carbohydrates, 275g. Each row is indexed by the recipe id from the original dataset, which is used to match the rows and drop the duplicate recipes.

| id | calories | total fat | sugar | sodium | protein | saturated fat | carbohydrates |
|-------:|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|
| 333281 |      138.4 |        6.5  |   12.5  |    0.069 |       1.5 |             3.8 |           16.5  |
| 453467 |      595.1 |       29.9  |   52.75 |    0.506 |       6.5 |            10.2 |           71.5  |
| 306168 |      194.8 |       13    |    1.5  |    0.736 |      11   |             7.2 |            8.25 |
| 286009 |      878.3 |       40.95 |   81.5  |    0.299 |      10   |            24.6 |          107.25 |
| 475785 |      267   |       19.5  |    3    |    0.276 |      14.5 |             9.6 |            5.5  |


## Baseline Model

To start, I will create a baseline model predicting calories based on grams of fat and grams of carbohydrates. I've chosen these to start because they are macronutrients and directly contribute to calories in food. One gram of fat has 9 calories and one gram of carbohydrates has 4 calories. Linear regression will be executed using sklearn, and the model's performance will be evaluated using RMSE. Both of the features are numeric features, and do not need to be engineered to be adequate for the linear regression model. Before fitting the linear regression model, the final cleaned dataframe was split into train and test sets (25% test split). Each set is split by features and response, with the features being stored in X_train and X_test, and the response (calories), being stored in y_train and y_test.

Code for sklearn Linear Regression pipeline:

    base_pl = Pipeline([
        ('lin-reg', LinearRegression())
    ])
    base_pl.fit(X_train[['total fat', 'carbohydrates']], y_train)

Using the fit model to predict the calories of the test dataset, and finding RMSE.

    y_test_pred = base_pl.predict(X_test[['total fat', 'carbohydrates']])
    rmse_base = np.sqrt(mean_squared_error(y_test, y_test_pred))

The RMSE of the test data for this model is 110. An RMSE of 110 for this prediction is bad. With the number of calories for many of these dishes hovering between 100 and 1000, this means that the model is consistently off by a considerable amount. This model will be improved in the next section.


## Final Model

In order to improve this model, I am going to add the rest of the nutrition features and see how much it improves the RMSE. I assume the only additional feature that will be useful in the prediciton is protein, but I will add all of them for due dilligence. Afterwards, I will run a Lasso on the new model, which will help me find and remove the features it deems to be irrelevant for predicting calories.

Code for new sklearn Linear Regression pipeline, predicting Calories from X_test, and finding RMSE:

    all_pl = Pipeline([
        ('lin-reg', LinearRegression())
    ])
    all_pl.fit(X_train, y_train)
    all_test_pred = all_pl.predict(X_test)
    rmse_all = np.sqrt(mean_squared_error(y_test, all_test_pred))
    
The RMSE of the linear regression model using all the nutrition facts is 54.18. The model significantly improved by adding more features. Now let's see if we need all of them to achieve the same performance

The Lasso is a variable selection method that works by squeezing the regression coefficients towards 0 during calculations. If the Lasso squeezes a coefficient all the way to 0, it has determined that the variable corresponding to that coefficient does not have a significant effect on the regression model. This helps identify which variables are important and which can be removed.

The lasso_path() function in sklearn's linear_model package allows for testing of multiple different alpha values, the hyperparameter in the Lasso controlling the shrinkage. Alpha needs to be tuned, and the values in the dataframe below are the lasso regression coefficients for each feature for each alpha value tested. I ran the function for alpha values between 0.5 and 4.5, at increments of 0.5. The results are shown below.

|               | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 |
|:--------------|-----------:|------------:|------------:|-----------:|------------:|------------:|------------:|-----------:|-----------:|
| total fat     |  8.84876   |  8.8485     |  8.84855    |  8.84853   |  8.8482     |  8.84807    |  8.84832    |  8.84858   |  8.84885   |
| sugar         | -0.0086413 | -0.00768376 | -0.00752075 | -0.0075236 | -0.00775076 | -0.00881055 | -0.00996079 | -0.0110539 | -0.0121478 |
| sodium        |  0         |  0          |  0          |  0         |  0.00295566 |  0.0448232  |  0.0872197  |  0.129704  |  0.172194  |
| protein       |  4.44884   |  4.45038    |  4.45116    |  4.45192   |  4.45283    |  4.45265    |  4.45208    |  4.45152   |  4.45096   |
| saturated fat |  0         |  0          |  0          |  0         |  0          |  0          |  0          | -0         | -0         |
| carbohydrates |  4.25084   |  4.2501     |  4.24992    |  4.24987   |  4.24999    |  4.25028    |  4.2506     |  4.25086   |  4.25113   |

The interpretation of the Lasso Regresssion across the tested alpha values is the same. Total Fat, Protein, and Carbohydrates are the three features that are the most relevant to predicting the number of calories. The other three features saw their coefficients shrink to 0 (or very close to 0), deeming them irrelevent. This makes sense, as these are the three macronutrients that make up food and provide calories. The coefficients also make sense, as fat has about 9 calories per gram, and protein and carbs each have about 4 calories per gram.

Using the variables the Lasso deemed to be the most important: Total Fat, Protein, and Carbohydrates, the Linear Regression model can be rerun for a third time with only these three features.

    select_pl = Pipeline([
        ('lin-reg', LinearRegression())
    ])
    select_pl.fit(X_train[['total fat', 'protein', 'carbohydrates']], y_train)
    select_preds = select_pl.predict(X_test[['total fat', 'protein', 'carbohydrates']])
    rmse_select = np.sqrt(mean_squared_error(y_test, select_preds))
    
The RMSE for this model came out to 54.1, virtually the same as the model with all 6 features. This is expected, as the three features are simply a subset of the 6 used for the previous model. However, without the extra variables, this new model is a better model because it is simpler. Simpler models with less features are preferred over more complicated models with more features.

This process can be replicated using the LassoCV function from sklearn.linear_model, which handles the cross validation and tuning for the hyperparameter alpha as well as the predictions for the test data. The same result is found, RMSE is also 54.1.

    las = linear_model.LassoCV(cv = 5, random_state = 42).fit(X_train, y_train)
    np.sqrt(mean_squared_error(y_test,las.predict(X_test)))
    
The final model isthe lasso regression model found using the LassoCV() method and saved in the variable las

## Fairness Analysis

