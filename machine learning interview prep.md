# Linear Regression

Linear regression is a fundamental technique in statistics and machine learning used to model the relationship between a dependent variable and one or more independent variables. Here's a comprehensive overview:

## 1. Concept of Linear Regression

**Linear Regression** aims to find the best-fitting linear relationship between a dependent variable \( y \) and one or more independent variables \( X \). The goal is to predict \( y \) using the linear combination of \( X \).

## 2. Types of Linear Regression

- **Simple Linear Regression**: Models the relationship between a single independent variable \( X \) and the dependent variable \( y \). The model can be represented as:
  \[
  y = \beta_0 + \beta_1 X + \epsilon
  \]
  where:
  - \( \beta_0 \) is the y-intercept.
  - \( \beta_1 \) is the slope of the line.
  - \( \epsilon \) is the error term.

- **Multiple Linear Regression**: Extends simple linear regression to multiple independent variables \( X_1, X_2, \ldots, X_p \). The model is represented as:
  \[
  y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p + \epsilon
  \]

## 3. Assumptions of Linear Regression

For linear regression to be valid, certain assumptions need to be met:

1. **Linearity**: The relationship between the independent and dependent variables should be linear.
2. **Independence**: The residuals (errors) should be independent of each other.
3. **Homoscedasticity**: The residuals should have constant variance at all levels of the independent variables.
4. **Normality**: The residuals should be normally distributed.

## 4. Fitting the Model

The process of fitting a linear regression model involves estimating the coefficients \( \beta_0, \beta_1, \ldots, \beta_p \) that minimize the sum of squared residuals (the differences between the observed and predicted values).

**Ordinary Least Squares (OLS)** is the most common method for estimating these coefficients. The goal of OLS is to minimize the sum of squared differences between the observed values and the values predicted by the linear model.

## 5. Evaluating the Model

- **Coefficient of Determination (\( R^2 \))**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates perfect prediction.
  
- **Adjusted \( R^2 \)**: Adjusts \( R^2 \) for the number of predictors in the model. It's useful when comparing models with different numbers of predictors.

- **Mean Squared Error (MSE)**: Average of the squares of the errors (residuals). It gives a measure of the average squared difference between observed and predicted values.

- **Root Mean Squared Error (RMSE)**: Square root of MSE. It provides a measure of the average magnitude of the errors in the same units as the dependent variable.

## 6. Implementation in Python

Here's a basic example of how to implement linear regression in Python using `scikit-learn`:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Example data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([2, 3, 5, 7, 11])            # Dependent variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a model and fit it
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the model
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
```


## Code for lR from scratch
```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.beta_0 = 0  # Intercept
        self.beta_1 = 0  # Slope

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        X (numpy array): Independent variable(s)
        y (numpy array): Dependent variable
        """
        # Compute the means of X and y
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        # Compute the slope (beta_1) and intercept (beta_0)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        self.beta_1 = numerator / denominator
        self.beta_0 = y_mean - (self.beta_1 * X_mean)

    def predict(self, X):
        """
        Predict the dependent variable for given independent variable(s).

        Parameters:
        X (numpy array): Independent variable(s)

        Returns:
        numpy array: Predicted values
        """
        return self.beta_0 + self.beta_1 * X

    def score(self, X, y):
        """
        Compute the R^2 score of the model.

        Parameters:
        X (numpy array): Independent variable(s)
        y (numpy array): Dependent variable

        Returns:
        float: R^2 score
        """
        y_pred = self.predict(X)
        total_variance = np.sum((y - np.mean(y)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y)) ** 2)
        return explained_variance / total_variance

# Example Usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(0)
    X = np.random.rand(100) * 10
    y = 2.5 * X + np.random.randn(100) * 2

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict values
    y_pred = model.predict(X)

    # Print the coefficients
    print(f"Intercept (beta_0): {model.beta_0}")
    print(f"Slope (beta_1): {model.beta_1}")

    # Evaluate the model
    r2_score = model.score(X, y)
    print(f"R^2 Score: {r2_score}")

    # Plot the results
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, y_pred, color='red', label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
```


# 15 Most Asked Linear Regression Interview Questions

## 1. What is linear regression?

**Answer**: Linear regression is a statistical method used to model and analyze the relationship between a dependent variable and one or more independent variables. The model assumes that there is a linear relationship between the variables.

## 2. What is the difference between simple and multiple linear regression?

**Answer**: Simple linear regression involves one independent variable and one dependent variable, whereas multiple linear regression involves two or more independent variables and one dependent variable.

## 3. What are the key assumptions of linear regression?

**Answer**: The key assumptions are:
- **Linearity**: The relationship between the dependent and independent variables is linear.
- **Independence**: Residuals are independent of each other.
- **Homoscedasticity**: Residuals have constant variance.
- **Normality**: Residuals are normally distributed.

## 4. What is the purpose of the coefficient of determination (R²)?

**Answer**: R² measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, where 1 indicates a perfect fit.

## 5. What is multicollinearity and how can you detect it?

**Answer**: Multicollinearity occurs when independent variables are highly correlated with each other, which can cause instability in the coefficient estimates. It can be detected using variance inflation factor (VIF) or correlation matrices.

## 6. What is the difference between adjusted R² and R²?

**Answer**: Adjusted R² adjusts R² for the number of predictors in the model. Unlike R², which always increases with more predictors, adjusted R² can decrease if additional predictors do not improve the model significantly.

## 7. What is the Ordinary Least Squares (OLS) method?

**Answer**: OLS is a method used to estimate the parameters of a linear regression model. It minimizes the sum of the squared residuals (the differences between observed and predicted values) to find the best-fitting line.

## 8. What are residuals in linear regression?

**Answer**: Residuals are the differences between the observed values and the predicted values from the regression model. They represent the error or deviation of the model's predictions from the actual data.

## 9. How do you handle outliers in linear regression?

**Answer**: Outliers can be handled by:
- Identifying them using diagnostic plots or statistical tests.
- Transforming the data (e.g., log transformation).
- Using robust regression techniques that are less sensitive to outliers.

## 10. What is heteroscedasticity and how can you detect it?

**Answer**: Heteroscedasticity occurs when the variance of residuals is not constant across all levels of the independent variable(s). It can be detected using residual plots or statistical tests such as the Breusch-Pagan test.

## 11. What is regularization in linear regression and why is it used?

**Answer**: Regularization techniques like Lasso (L1 regularization) and Ridge (L2 regularization) add a penalty to the size of the coefficients to prevent overfitting and handle multicollinearity by shrinking the coefficients.

## 12. How can you interpret the coefficients in a multiple linear regression model?

**Answer**: Each coefficient represents the change in the dependent variable for a one-unit change in the corresponding independent variable, holding all other variables constant.

## 13. What is the difference between a parametric and a non-parametric model?

**Answer**: Parametric models assume a specific functional form for the relationship between variables (e.g., linear regression assumes a linear relationship). Non-parametric models do not assume a specific functional form and can model more complex relationships.

## 14. How can you assess the goodness-of-fit of a linear regression model?

**Answer**: Goodness-of-fit can be assessed using metrics such as R², adjusted R², Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and residual analysis.

## 15. Explain the concept of feature scaling and why it is important in linear regression.

**Answer**: Feature scaling involves normalizing or standardizing the independent variables so they have a similar scale. This is important in linear regression (especially with regularization) to ensure that all features contribute equally to the model and to improve the stability and performance of the model.


# Advantages and Disadvantages of Linear Regression

## Advantages

1. **Simplicity**:
   - Linear regression is easy to understand and interpret. The relationship between variables is modeled as a straight line, making it straightforward to explain and visualize.

2. **Computational Efficiency**:
   - It is computationally efficient and can be implemented quickly. It doesn’t require complex calculations, making it suitable for large datasets.

3. **No Need for Feature Scaling**:
   - Linear regression generally doesn’t require feature scaling or normalization of variables (though it can help with numerical stability, especially in regularized models).

4. **Flexibility**:
   - It can be extended to multiple linear regression, allowing it to handle multiple independent variables.

5. **Foundation for More Complex Models**:
   - Linear regression serves as a fundamental technique that underpins more complex regression methods and models, such as polynomial regression or regularized regression techniques (Ridge, Lasso).

6. **Easy to Implement**:
   - Many libraries and tools (like `scikit-learn` in Python) provide built-in functions for implementing linear regression, making it accessible for practitioners.

7. **Predictive Power**:
   - With appropriate assumptions and data quality, linear regression can provide strong predictive power for many real-world applications.

## Disadvantages

1. **Assumption of Linearity**:
   - Linear regression assumes a linear relationship between the dependent and independent variables. This may not hold true for complex or non-linear relationships.

2. **Sensitivity to Outliers**:
   - The model can be highly sensitive to outliers, which can disproportionately affect the slope and intercept of the fitted line.

3. **Multicollinearity**:
   - When independent variables are highly correlated (multicollinearity), it can lead to unreliable estimates of the coefficients and affect the model's interpretability.

4. **Assumption of Homoscedasticity**:
   - Linear regression assumes that the residuals (errors) have constant variance (homoscedasticity). In practice, this may not always be the case, leading to inefficient or biased estimates.

5. **Limited to Linear Relationships**:
   - Linear regression is not suitable for modeling complex relationships unless extended (e.g., polynomial regression), and it may fail to capture intricate patterns in the data.

6. **Normality of Residuals**:
   - Linear regression assumes that residuals are normally distributed. If this assumption is violated, it may affect the validity of statistical tests and confidence intervals.

7. **Overfitting with Multiple Predictors**:
   - With too many predictors, especially without proper regularization, the model may overfit the training data, resulting in poor generalization to new data.

8. **Multivariate Outliers**:
   - Linear regression may not handle multivariate outliers well, which can skew the results and affect model performance.

# logistic regression
