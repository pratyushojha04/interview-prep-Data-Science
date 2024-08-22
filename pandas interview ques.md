# Important Pandas Interview Questions

## 1. What is Pandas and why is it used in Python?
Pandas is a powerful data manipulation and analysis library for Python, providing data structures like DataFrame and Series for efficient data handling and analysis.

## 2. How do you create a DataFrame and a Series in Pandas?
Use `pd.DataFrame()` to create a DataFrame and `pd.Series()` to create a Series.

## 3. How do you handle missing data in Pandas DataFrames?
Use methods like `dropna()`, `fillna()`, and `isna()` to handle missing data by dropping, filling, or identifying NaN values.

## 4. How can you perform data filtering in a Pandas DataFrame?
Use boolean indexing or methods like `query()` to filter rows based on conditions.

## 5. How do you merge, join, and concatenate DataFrames in Pandas?
Use `pd.merge()`, `pd.concat()`, and DataFrame methods like `join()` to combine DataFrames based on common columns or indices.

## 6. What is the difference between `loc[]` and `iloc[]` in Pandas?
`loc[]` is label-based indexing, while `iloc[]` is integer-location based indexing.

## 7. How can you group data in a Pandas DataFrame and apply aggregate functions?
Use `groupby()` to group data and apply aggregate functions like `sum()`, `mean()`, `count()`, etc.

## 8. How do you handle categorical data in Pandas?
Use methods like `pd.get_dummies()` for one-hot encoding or `pd.Categorical()` for categorical data types.

## 9. How do you perform data aggregation and pivoting in Pandas?
Use `pivot_table()` and `agg()` for data aggregation and pivoting operations.

## 10. How can you sort a DataFrame by one or more columns?
Use the `sort_values()` method to sort a DataFrame by one or more columns.

## 11. How do you apply functions to DataFrame columns or rows?
Use the `apply()` method to apply a function to columns or rows of a DataFrame.

## 12. How can you handle duplicate data in a Pandas DataFrame?
Use the `drop_duplicates()` method to remove duplicate rows.

## 13. What are the methods to handle time series data in Pandas?
Use `pd.to_datetime()`, `resample()`, `shift()`, and `rolling()` to handle and analyze time series data.

## 14. How do you read and write data to various file formats using Pandas?
Use `pd.read_csv()`, `pd.read_excel()`, `pd.read_json()`, and corresponding `to_` methods like `to_csv()`, `to_excel()`, `to_json()` for file operations.

## 15. How can you perform element-wise operations on DataFrames?
Use arithmetic operators or methods like `add()`, `sub()`, `mul()`, and `div()` for element-wise operations.

## 16. How do you perform hierarchical indexing (MultiIndex) in Pandas?
Use `pd.MultiIndex.from_tuples()` or `set_index()` with multiple columns to create hierarchical indices.

## 17. How can you use the `applymap()` method in Pandas?
`applymap()` applies a function to each element of a DataFrame.

## 18. How do you manipulate DataFrame indices and columns?
Use methods like `set_index()`, `reset_index()`, and `rename()` to manipulate indices and columns.

## 19. How can you visualize data using Pandas?
Use the `plot()` method or integrate with libraries like Matplotlib or Seaborn for visualization.

## 20. How do you handle large datasets that do not fit into memory using Pandas?
Use chunking with `pd.read_csv()` or Dask to work with large datasets.

## 21. How do you use the `pivot_table()` method in Pandas?
`pivot_table()` creates a pivot table based on specified index, columns, and aggregation functions.

## 22. How can you perform string operations on DataFrame columns?
Use string methods like `str.contains()`, `str.replace()`, and `str.split()` on DataFrame columns.

## 23. How do you perform element-wise comparison between DataFrames?
Use comparison operators like `==`, `!=`, `<`, `>`, `<=`, `>=`, and functions like `eq()`, `ne()`, etc.

## 24. How do you handle and analyze missing values in time series data?
Use methods like `resample()` to handle missing values by resampling and filling gaps.

## 25. How can you use the `transform()` method in Pandas?
`transform()` applies a function to each group of a DataFrame or Series and returns a DataFrame with the same shape.

## 26. How do you apply multiple functions to DataFrame columns?
Use the `agg()` method to apply multiple functions to DataFrame columns.

## 27. How do you use the `combine_first()` method in Pandas?
`combine_first()` combines two DataFrames, with missing values in the first DataFrame filled by values from the second DataFrame.

## 28. How do you handle categorical data and apply one-hot encoding?
Use `pd.get_dummies()` to apply one-hot encoding to categorical data.

## 29. How do you perform cross-tabulations and contingency tables in Pandas?
Use `pd.crosstab()` to create cross-tabulations and contingency tables.

## 30. How can you calculate rolling statistics in Pandas?
Use the `rolling()` method followed by aggregation functions like `mean()`, `sum()`, etc., to calculate rolling statistics.

## 31. How do you calculate correlation and covariance matrices in Pandas?
Use `df.corr()` for correlation and `df.cov()` for covariance matrices.

## 32. How do you handle outliers in a DataFrame?
Identify and handle outliers using methods like filtering, capping, or transforming data based on statistical measures.

## 33. How do you perform operations on DataFrame columns with different data types?
Use methods and operations specific to the data types of the columns, ensuring compatibility and appropriate handling.

## 34. How do you use the `resample()` method in Pandas for time series data?
`resample()` is used to resample time series data to different frequencies, such as daily to monthly.

## 35. How do you use `pd.concat()` to concatenate DataFrames along different axes?
Use `pd.concat()` with the `axis` argument to concatenate DataFrames along rows or columns.

## 36. How do you compute and interpret summary statistics in Pandas?
Use methods like `describe()`, `mean()`, `median()`, `std()`, etc., to compute and interpret summary statistics.

## 37. How do you filter rows in a DataFrame based on conditions in multiple columns?
Use boolean indexing with conditions combined using `&` (and) or `|` (or) operators.

## 38. How can you use `groupby()` with multiple columns for aggregation?
Pass a list of columns to `groupby()` to group data by multiple columns and apply aggregation functions.

## 39. How do you use the `sample()` method to draw random samples from a DataFrame?
Use `sample()` to draw random samples with specified size and optional parameters like `replace` and `weights`.

## 40. How do you handle duplicate data based on specific columns?
Use `drop_duplicates(subset=...)` to remove duplicates based on specific columns.

## 41. How do you perform conditional replacement in a DataFrame?
Use `np.where()` or the `apply()` method with custom functions to conditionally replace values.

## 42. How do you use `pd.merge()` to combine DataFrames with different join types?
Specify `how='inner'`, `how='outer'`, `how='left'`, or `how='right'` in `pd.merge()` to perform different types of joins.

## 43. How can you handle date and time data with Pandas?
Use `pd.to_datetime()` to convert strings to datetime objects and manipulate date and time data.

## 44. How do you perform data manipulation and feature engineering on DataFrames?
Use methods like `apply()`, `map()`, and `transform()` for feature engineering and data manipulation.

## 45. How do you use the `cut()` method for binning continuous data?
`cut()` bins continuous data into discrete intervals or bins.

## 46. How do you calculate and plot the distribution of data in a DataFrame?
Use `plot()` with histograms or KDE plots to visualize the distribution of data.

## 47. How do you perform group-wise operations using `groupby()` and `agg()`?
Use `groupby()` to group data and `agg()` to apply multiple aggregation functions to each group.

## 48. How do you handle multi-index DataFrames?
Use methods like `stack()`, `unstack()`, and `reset_index()` to handle multi-index DataFrames.

## 49. How can you perform time-based resampling and frequency conversion?
Use `resample()` to resample data to different frequencies and handle time-based conversions.

## 50. How do you use `pd.pivot_table()` to summarize data?
`pd.pivot_table()` creates a pivot table to summarize and aggregate data based on specified index and columns.
# Additional Important Pandas Interview Questions

## 51. How do you handle large datasets that exceed memory capacity using Pandas?
Use chunking with `pd.read_csv()` and Dask to process large datasets in smaller, manageable pieces.

## 52. How do you use `pd.Series.map()` to transform data in a Series?
`map()` applies a function or a dictionary of replacements to each element in a Series.

## 53. How do you identify and handle outliers in time series data?
Use statistical methods like Z-score or IQR to detect and handle outliers in time series data.

## 54. How do you perform operations on DataFrame columns with different data types?
Apply methods and operations appropriate to the data type, such as numerical or categorical operations.

## 55. How can you use `pd.get_dummies()` to convert categorical variables into dummy/indicator variables?
`pd.get_dummies()` creates binary columns for each category in categorical data.

## 56. How do you use the `apply()` method for element-wise operations on DataFrame or Series?
`apply()` allows you to apply a function along an axis of a DataFrame or to elements of a Series.

## 57. How do you perform and interpret time-based rolling statistics?
Use `rolling()` with aggregation functions like `mean()`, `sum()`, or `std()` to compute rolling statistics.

## 58. How can you perform and interpret cross-tabulations and contingency tables?
Use `pd.crosstab()` to compute and analyze frequency distributions of categorical variables.

## 59. How do you use `pd.merge_asof()` for merging time-series data?
`pd.merge_asof()` performs an asof merge, combining data based on nearest keys and ordering.

## 60. How can you use the `transform()` method to perform element-wise transformations on grouped data?
`transform()` applies a function to each group, returning a DataFrame with the same shape.

## 61. How do you perform and interpret feature scaling and normalization in Pandas?
Use methods like Min-Max scaling or Standardization (Z-score normalization) to scale and normalize data.

## 62. How do you handle multi-level indexing (MultiIndex) and perform operations on it?
Use `set_index()`, `reset_index()`, and `stack()`/`unstack()` methods to manipulate and analyze MultiIndex DataFrames.

## 63. How do you use the `cut()` method for discretizing continuous data into bins?
`cut()` bins continuous data into discrete intervals, allowing for categorical analysis.

## 64. How do you perform aggregation and transformation operations with `groupby()`?
Use `groupby()` in combination with `agg()`, `transform()`, and `apply()` to aggregate and transform grouped data.

## 65. How do you handle missing values in time series data with resampling?
Use `resample()` to fill in missing values and handle gaps in time series data.

## 66. How can you apply custom functions to DataFrame rows or columns using `apply()`?
Use `apply()` with a custom function to perform complex transformations or calculations on DataFrame rows or columns.

## 67. How do you use `pd.concat()` to concatenate DataFrames with different columns?
Use `pd.concat()` with `axis=0` for row-wise concatenation or `axis=1` for column-wise concatenation.

## 68. How do you use the `combine_first()` method to handle missing data between DataFrames?
`combine_first()` fills missing values in one DataFrame with values from another DataFrame.

## 69. How do you perform and interpret operations on data with `DataFrame.applymap()`?
`applymap()` applies a function to each element of a DataFrame, useful for element-wise operations.

## 70. How do you use `pd.pivot()` to reshape data from long to wide format?
`pd.pivot()` reshapes DataFrames from a long format to a wide format by specifying index, columns, and values.

## 71. How do you use `pd.cut()` for binning and segmenting numerical data?
`pd.cut()` segments numerical data into bins and returns categorical data.

## 72. How do you handle hierarchical index (MultiIndex) DataFrames in groupby operations?
Use `groupby()` on MultiIndex DataFrames to perform grouped operations based on multiple levels.

## 73. How do you compute and interpret statistical measures like skewness and kurtosis in Pandas?
Use `df.skew()` and `df.kurt()` to calculate and interpret skewness and kurtosis.

## 74. How do you handle and analyze large datasets using Dask in conjunction with Pandas?
Use Dask to parallelize and scale data processing tasks beyond memory limits, integrating with Pandas for analysis.

## 75. How do you perform and interpret time-based resampling with `pd.Grouper()`?
Use `pd.Grouper()` with `resample()` to group time series data by specified time intervals.

## 76. How do you handle complex time series operations such as shifting and lagging?
Use `shift()` to create lagged features and `shift()` combined with `resample()` for time-based shifting.

## 77. How do you use `pd.DataFrame.rename()` to rename columns or indices?
`rename()` allows you to rename columns or indices using a dictionary or function.

## 78. How do you use `pd.Series.value_counts()` to count occurrences of unique values?
`value_counts()` returns a Series containing counts of unique values in a Series.

## 79. How can you use `pd.Series.rolling()` to compute rolling window statistics?
`rolling()` computes statistics like mean, sum, and std over a rolling window of specified size.

## 80. How do you perform and interpret operations on hierarchical indexed data?
Use `stack()`, `unstack()`, and `swaplevel()` to manipulate and analyze data with hierarchical indexing.

## 81. How do you handle categorical data with `Categorical` data type in Pandas?
Use `pd.Categorical()` to define categorical data types, facilitating efficient operations on categorical variables.

## 82. How do you perform data transformation and feature extraction using `apply()`?
Use `apply()` to perform custom data transformations and extract features from DataFrame columns or rows.

## 83. How do you handle and analyze duplicate data based on specific conditions?
Use `drop_duplicates()` with the `subset` parameter to handle duplicates based on specific columns.

## 84. How do you use `pd.Series.str` methods for string manipulation in a Series?
Use `str` accessor methods like `str.contains()`, `str.replace()`, and `str.split()` for string operations.

## 85. How do you perform conditional data replacement using Pandas?
Use `np.where()` or conditional indexing to replace data based on specified conditions.

## 86. How do you compute and interpret cumulative statistics in a DataFrame?
Use `cumsum()`, `cumprod()`, and similar methods to compute cumulative sums, products, etc.

## 87. How do you handle time zones and convert time series data across different time zones?
Use `tz_localize()` and `tz_convert()` to manage and convert time series data across time zones.

## 88. How do you perform complex queries and data selection with `query()` method?
Use `query()` to perform complex data queries using a string expression for filtering rows.

## 89. How do you handle out-of-memory issues when working with large datasets?
Use chunking methods, Dask, or other out-of-core computing techniques to handle large datasets efficiently.

## 90. How do you use `pd.merge()` with multiple keys for more complex joins?
Pass multiple columns to `on` parameter in `pd.merge()` to join DataFrames on multiple keys.

## 91. How do you use `pd.Series.interpolate()` to fill missing values in a Series?
`interpolate()` fills missing values using interpolation methods like linear, polynomial, etc.

## 92. How do you perform and interpret data aggregation with `pivot_table()` in Pandas?
Use `pivot_table()` to aggregate data and compute summary statistics across specified dimensions.

## 93. How do you handle data type conversion in Pandas DataFrames?
Use methods like `astype()` to convert data types of columns or entire DataFrames.

## 94. How do you use `pd.Series.idxmax()` and `pd.Series.idxmin()` for index-based operations?
`idxmax()` and `idxmin()` return the index of the maximum and minimum values in a Series, respectively.

## 95. How do you perform and interpret window functions like moving averages with Pandas?
Use `rolling()` followed by aggregation functions to compute moving averages and other window functions.

## 96. How do you use `pd.Series.apply()` to apply functions to elements in a Series?
`apply()` applies a function to each element of a Series, allowing for complex transformations.

## 97. How do you use `pd.DataFrame.sample()` to randomly sample rows from a DataFrame?
`sample()` allows you to randomly sample rows from a DataFrame with specified size and parameters.

## 98. How do you use `pd.Series.value_counts()` to get counts of unique values in a Series?
`value_counts()` returns a Series of counts of unique values in a Series.

## 99. How do you use `pd.DataFrame.groupby()` with aggregation functions like `sum()`, `mean()`, and `count()`?
Use `groupby()` in combination with aggregation functions to perform group-wise operations and compute summaries.

## 100. How do you use `pd.Series.to_csv()` and `pd.DataFrame.to_csv()` to write Series and DataFrames to CSV files?
Use `to_csv()` to write Series or DataFrame objects to CSV files for storage or further analysis.
# Additional Important Pandas Interview Questions

## 101. How do you handle and perform operations on large datasets that don't fit into memory?
Use techniques such as chunking with `pd.read_csv()` or leveraging libraries like Dask for out-of-core computations.

## 102. How do you use `pd.DataFrame.pipe()` to apply functions in a pipeline?
`pipe()` allows you to apply a function to a DataFrame in a pipeline, facilitating cleaner and more readable code.

## 103. How do you work with multi-dimensional data using `pd.Panel`?
Note: `pd.Panel` is deprecated. Instead, use MultiIndex DataFrames or xarray for multi-dimensional data.

## 104. How do you perform and interpret data manipulation operations with the `DataFrame.query()` method?
`query()` allows for efficient querying of DataFrame rows using a string expression, simplifying data selection.

## 105. How do you use `pd.DataFrame.mask()` and `pd.DataFrame.where()` for conditional replacement?
`mask()` replaces values where the condition is `True`, while `where()` replaces values where the condition is `False`.

## 106. How do you merge DataFrames with overlapping column names using `pd.merge()`?
Handle overlapping column names by using the `suffixes` parameter to differentiate between columns from different DataFrames.

## 107. How do you use `pd.Series.rank()` to rank data in a Series?
`rank()` assigns ranks to data in a Series, useful for ranking and comparing values.

## 108. How do you use `pd.DataFrame.assign()` to add or modify columns in a DataFrame?
`assign()` allows you to add or modify columns in a DataFrame by specifying new column names and values.

## 109. How do you use `pd.Series.shift()` to perform time-based operations such as lagging and leading?
`shift()` allows you to shift data in a Series or DataFrame, useful for creating lagged features or aligning data.

## 110. How do you use `pd.DataFrame.pivot_table()` to create pivot tables and compute aggregated statistics?
`pivot_table()` creates pivot tables that summarize data by aggregating values across specified dimensions.

## 111. How do you use `pd.Series.map()` to transform values in a Series using a mapping function or dictionary?
`map()` applies a function or dictionary to transform values in a Series.

## 112. How do you handle time series data with `pd.date_range()` and `pd.to_datetime()`?
Use `pd.date_range()` to generate a sequence of dates and `pd.to_datetime()` to convert strings or other formats to datetime objects.

## 113. How do you handle and analyze data with missing or duplicated values using Pandas?
Use methods like `dropna()`, `fillna()`, and `drop_duplicates()` to handle missing or duplicated data.

## 114. How do you use `pd.DataFrame.corr()` to compute correlation matrices between columns?
`corr()` computes pairwise correlation of columns, useful for understanding relationships between variables.

## 115. How do you use `pd.Series.str.contains()` to filter text data based on a substring?
`str.contains()` allows you to filter Series based on whether elements contain a specified substring.

## 116. How do you use `pd.Series.rolling()` to compute rolling statistics with various window sizes?
`rolling()` computes rolling statistics such as moving averages over a specified window size.

## 117. How do you use `pd.DataFrame.explode()` to transform lists of values into separate rows?
`explode()` expands lists or arrays in a DataFrame into separate rows, useful for normalizing nested data.

## 118. How do you use `pd.Series.value_counts()` to get the frequency of unique values in a Series?
`value_counts()` returns a Series containing counts of unique values, useful for frequency analysis.

## 119. How do you use `pd.Series.unique()` to get unique values in a Series?
`unique()` returns an array of unique values in a Series, useful for understanding the distinct values in a dataset.

## 120. How do you use `pd.Series.to_frame()` to convert a Series to a DataFrame?
`to_frame()` converts a Series to a DataFrame, allowing you to perform DataFrame operations on Series data.

## 121. How do you use `pd.DataFrame.T` to transpose a DataFrame?
`T` transposes the DataFrame, swapping rows and columns, which can be useful for certain operations and visualizations.

## 122. How do you handle data normalization and standardization in Pandas?
Use methods like `(df - df.mean()) / df.std()` for standardization and `(df - df.min()) / (df.max() - df.min())` for normalization.

## 123. How do you handle categorical data with ordered categories in Pandas?
Use `pd.Categorical()` with the `categories` and `ordered` parameters to define and handle ordered categorical data.

## 124. How do you use `pd.Series.apply()` to apply a function along an axis of a DataFrame?
`apply()` allows you to apply a function along a specified axis of a DataFrame, useful for complex transformations.

## 125. How do you use `pd.DataFrame.to_sql()` to write DataFrame data to an SQL database?
`to_sql()` writes DataFrame data to a SQL database, useful for integrating Pandas with SQL databases.

## 126. How do you use `pd.DataFrame.head()` and `pd.DataFrame.tail()` to preview data?
`head()` and `tail()` allow you to view the first and last few rows of a DataFrame, respectively, for quick data inspection.

## 127. How do you use `pd.Series.diff()` to compute the difference between consecutive elements in a Series?
`diff()` computes the difference between consecutive elements, useful for analyzing changes over time.

## 128. How do you use `pd.Series.describe()` to get summary statistics of a Series?
`describe()` provides a summary of statistics such as mean, median, and standard deviation for a Series.

## 129. How do you use `pd.DataFrame.pivot()` to reshape data with multiple values?
`pivot()` reshapes data from long to wide format by specifying index, columns, and values, useful for data analysis.

## 130. How do you handle time series data with missing time points using Pandas?
Use `reindex()` with `pd.date_range()` to fill missing time points and align time series data.

# some examples to understand and get better in  pandas
# Practical Pandas Interview Questions

## 1. Data Cleaning and Analysis
You have a dataset of customer transactions with columns `CustomerID`, `TransactionDate`, `Amount`, and `Category`. Perform the following tasks:
- Load the data and handle missing values.
- Convert the `TransactionDate` column to datetime format.
- Calculate the total amount spent by each customer.
- Identify the top 5 categories by total spending.
- Create a new column indicating whether a transaction amount is above or below the median amount.
- Generate a pivot table showing total spending by `CustomerID` and `Category`.

## 2. Time Series Analysis
You are given a time series dataset of daily temperatures with columns `Date` and `Temperature`. Complete the following:
- Load the data and ensure the `Date` column is in datetime format.
- Set the `Date` column as the index of the DataFrame.
- Resample the data to monthly frequency, calculating the average temperature for each month.
- Plot the time series data and the monthly average temperatures using matplotlib.
- Handle any missing temperature values by forward filling.

## 3. Data Aggregation and Transformation
You have a dataset containing sales data with columns `Date`, `ProductID`, `SalesAmount`, and `Region`. Perform these tasks:
- Load the dataset and convert the `Date` column to datetime format.
- Create a new column indicating the year and month of each sale.
- Calculate the total sales amount for each `ProductID` and `Region`.
- Compute the percentage contribution of each `ProductID` to the total sales amount within each `Region`.
- Filter out regions where the total sales amount is less than $10,000.

## 4. Data Merging and Analysis
You have two DataFrames: `df1` with columns `EmployeeID`, `Name`, and `Department`, and `df2` with columns `EmployeeID`, `Salary`, and `HireDate`. Complete the following:
- Merge the two DataFrames on `EmployeeID`.
- Convert the `HireDate` column to datetime format.
- Calculate the tenure (in years) for each employee from their hire date to today.
- Identify the department with the highest average salary.
- Filter employees with a tenure of more than 5 years and a salary above the median salary.

## 5. Feature Engineering
You are working on a dataset with `ProductID`, `SaleDate`, `QuantitySold`, and `Revenue`. Perform these tasks:
- Load the data and convert `SaleDate` to datetime format.
- Create features for `Month`, `Quarter`, and `Year` from `SaleDate`.
- Calculate the average `QuantitySold` and `Revenue` per month.
- Create a new feature for the total sales (`QuantitySold * Revenue`) and add it to the DataFrame.
- Normalize the `Revenue` column to a range of [0, 1] and add it to the DataFrame.

## 6. Advanced Data Aggregation
You are given a dataset of employee performance with columns `EmployeeID`, `EvaluationDate`, `Score`, and `Department`. Complete the following:
- Load the data and convert `EvaluationDate` to datetime format.
- Calculate the average score for each `Department`.
- Identify the top 3 employees with the highest average score across all evaluations.
- Compute the rolling average score for each employee with a window size of 3 evaluations.
- Visualize the score trends for the top 3 employees over time.

## 7. Handling Multi-Index Data
You have a dataset with columns `Year`, `Quarter`, `ProductID`, `Sales` with hierarchical indexing on `Year` and `Quarter`. Perform these tasks:
- Load the data and set a multi-index on `Year` and `Quarter`.
- Calculate the total sales for each `Year` across all quarters.
- Unstack the `Quarter` level of the index to create a wide-format DataFrame.
- Calculate the percentage change in sales from one quarter to the next for each `ProductID`.
- Filter the data to show only products with a quarterly sales increase.

## 8. Data Visualization and Analysis
You have a dataset containing user activity logs with columns `UserID`, `ActivityDate`, `ActivityType`, and `Duration`. Complete the following:
- Load the data and convert `ActivityDate` to datetime format.
- Group the data by `UserID` and `ActivityType` and calculate the total `Duration` for each combination.
- Plot the total duration of each activity type for each user using a bar chart.
- Identify the user with the highest average activity duration.
- Generate a time series plot showing the daily total activity duration.

## 9. Data Transformation and Pivoting
You have a dataset with columns `StudentID`, `CourseID`, `Score`, and `Semester`. Perform these tasks:
- Load the data and create a pivot table with `StudentID` as rows, `CourseID` as columns, and `Score` as values.
- Compute the average score for each student across all courses.
- Normalize the scores within each course to a range of [0, 1].
- Identify students who scored above the 75th percentile in at least one course.
- Reshape the pivot table to show the average score for each course across semesters.

## 10. Advanced Time Series Forecasting
You are given a dataset with columns `Date` and `Sales`. Complete the following:
- Load the data and convert `Date` to datetime format.
- Set the `Date` column as the index and ensure it has no missing values.
- Decompose the time series into trend, seasonal, and residual components.
- Fit a simple exponential smoothing model to forecast the next 12 periods.
- Plot the historical sales and forecasted values on the same chart.


# some senario based problems with answer
# Scenario-Based Pandas Questions

## 1. Scenario: Sales Data Analysis
**Question:** You have a dataset of monthly sales data for different stores. The dataset includes `StoreID`, `Month`, `SalesAmount`, and `Region`. Perform the following tasks:
- Load the data into a DataFrame.
- Find the store with the highest average monthly sales.
- Identify the region with the highest total sales.
- Create a new column that shows the percentage change in sales from the previous month for each store.

**Answer:**
```python
import pandas as pd

# Load data
data = pd.read_csv('sales_data.csv')

# Find store with the highest average monthly sales
data['Month'] = pd.to_datetime(data['Month'])
monthly_avg_sales = data.groupby('StoreID')['SalesAmount'].mean()
highest_avg_store = monthly_avg_sales.idxmax()

# Identify region with the highest total sales
total_sales_by_region = data.groupby('Region')['SalesAmount'].sum()
highest_sales_region = total_sales_by_region.idxmax()

# Calculate percentage change in sales
data = data.sort_values(by=['StoreID', 'Month'])
data['PercentageChange'] = data.groupby('StoreID')['SalesAmount'].pct_change() * 100

highest_avg_store, highest_sales_region, data.head()
```


### 2. Scenario: Employee Performance Tracking
Question: You have a dataset containing employee performance reviews with EmployeeID, ReviewDate, PerformanceScore, and Department. The dataset spans several years.
2. Scenario: Employee Performance Tracking
Question: You have a dataset containing employee performance reviews with EmployeeID, ReviewDate, PerformanceScore, and Department. The dataset spans several years.

Load the data and convert ReviewDate to datetime format.
Calculate the average performance score for each department per year.
Identify the employee with the highest average performance score in each department.
Plot the performance scores for each department over time.

Load the data and convert ReviewDate to datetime format.
Calculate the average performance score for each department per year.
Identify the employee with the highest average performance score in each department.
Plot the performance scores for each department over time.


```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('performance_reviews.csv')

# Convert ReviewDate to datetime
data['ReviewDate'] = pd.to_datetime(data['ReviewDate'])

# Calculate average performance score for each department per year
data['Year'] = data['ReviewDate'].dt.year
average_score_per_department = data.groupby(['Year', 'Department'])['PerformanceScore'].mean().unstack()

# Identify employee with the highest average performance score in each department
average_score_per_employee = data.groupby('EmployeeID')['PerformanceScore'].mean()
best_employees = data.groupby('Department')['EmployeeID'].apply(lambda x: average_score_per_employee.loc[x].idxmax())

# Plot performance scores over time for each department
for department in data['Department'].unique():
    department_data = data[data['Department'] == department]
    department_scores = department_data.groupby('Year')['PerformanceScore'].mean()
    plt.plot(department_scores.index, department_scores.values, label=department)
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('performance_reviews.csv')

# Convert ReviewDate to datetime
data['ReviewDate'] = pd.to_datetime(data['ReviewDate'])

# Calculate average performance score for each department per year
data['Year'] = data['ReviewDate'].dt.year
average_score_per_department = data.groupby(['Year', 'Department'])['PerformanceScore'].mean().unstack()

# Identify employee with the highest average performance score in each department
average_score_per_employee = data.groupby('EmployeeID')['PerformanceScore'].mean()
best_employees = data.groupby('Department')['EmployeeID'].apply(lambda x: average_score_per_employee.loc[x].idxmax())

# Plot performance scores over time for each department
for department in data['Department'].unique():
    department_data = data[data['Department'] == department]
    department_scores = department_data.groupby('Year')['PerformanceScore'].mean()
    plt.plot(department_scores.index, department_scores.values, label=department)

plt.xlabel('Year')
plt.ylabel('Average Performance Score')
plt.legend()
plt.title('Performance Scores by Department Over Time')
plt.show()

best_employees, average_score_per_department.head()

plt.xlabel('Year')
plt.ylabel('Average Performance Score')
plt.legend()
plt.title('Performance Scores by Department Over Time')
plt.show()

best_employees, average_score_per_department.head()


```


### 3. Scenario: Customer Segmentation
Question: You have a dataset of customer transactions with CustomerID, TransactionDate, TransactionAmount, and ProductCategory.

Load the data and convert TransactionDate to datetime format.
Determine the total spending for each customer.
Segment customers into High, Medium, and Low spenders based on their total spending.
Create a summary table showing the average transaction amount by ProductCategory for each customer segment.
Answer:


```python
import pandas as pd

# Load data
data = pd.read_csv('customer_transactions.csv')

# Convert TransactionDate to datetime
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])

# Determine total spending for each customer
total_spending = data.groupby('CustomerID')['TransactionAmount'].sum()

# Segment customers
bins = [0, 100, 500, float('inf')]
labels = ['Low', 'Medium', 'High']
total_spending_category = pd.cut(total_spending, bins=bins, labels=labels)

# Create a summary table
data = data.copy()
data['CustomerSegment'] = data['CustomerID'].map(total_spending_category)
summary_table = data.groupby(['CustomerSegment', 'ProductCategory'])['TransactionAmount'].mean().unstack()

summary_table

```



### 4. Scenario: Product Demand Forecasting
Question: You have historical data of daily product sales with ProductID, Date, and SalesQuantity.

Load the data and convert Date to datetime format.
Resample the data to weekly frequency and calculate the total sales quantity for each product.
Create a rolling average of the sales quantity with a window size of 4 weeks.
Forecast the next 4 weeks of sales for each product using a simple linear regression model.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('product_sales.csv')

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Resample to weekly frequency
data.set_index('Date', inplace=True)
weekly_sales = data.groupby('ProductID').resample('W').agg({'SalesQuantity': 'sum'}).reset_index()

# Create rolling average
weekly_sales['RollingAverage'] = weekly_sales.groupby('ProductID')['SalesQuantity'].rolling(window=4).mean().reset_index(level=0, drop=True)

# Forecast next 4 weeks
forecasts = {}
for product in weekly_sales['ProductID'].unique():
    product_data = weekly_sales[weekly_sales['ProductID'] == product]
    product_data['WeekNumber'] = np.arange(len(product_data))
    
    # Prepare data for linear regression
    X = product_data[['WeekNumber']]
    y = product_data['SalesQuantity']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast next 4 weeks
    future_weeks = np.arange(len(product_data), len(product_data) + 4).reshape(-1, 1)
    forecast = model.predict(future_weeks)
    
    forecasts[product] = forecast

# Plot forecast for each product
for product, forecast in forecasts.items():
    plt.plot(range(len(forecast)), forecast, label=f'Product {product}')

plt.xlabel('Week')
plt.ylabel('Forecasted Sales Quantity')
plt.legend()
plt.title('Sales Forecast for Each Product')
plt.show()

forecasts
```
### 5. Scenario: Employee Attendance Tracking
Question: You have an employee attendance dataset with columns EmployeeID, Date, and AttendanceStatus (e.g., Present, Absent, Late).

Load the data and convert Date to datetime format.
Calculate the attendance rate for each employee over the last 6 months.
Identify employees with an attendance rate below 80%.
Create a summary of the number of days each employee was late in the last 6 months.

```python
import pandas as pd
from datetime import datetime, timedelta

# Load data
data = pd.read_csv('attendance.csv')

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the last 6 months
six_months_ago = datetime.now() - timedelta(days=180)
filtered_data = data[data['Date'] >= six_months_ago]

# Calculate attendance rate
total_days = filtered_data.groupby('EmployeeID')['Date'].count()
present_days = filtered_data[filtered_data['AttendanceStatus'] == 'Present'].groupby('EmployeeID')['Date'].count()
attendance_rate = (present_days / total_days) * 100

# Identify employees with an attendance rate below 80%
low_attendance_employees = attendance_rate[attendance_rate < 80].index

# Summary of late days
late_days_summary = filtered_data[filtered_data['AttendanceStatus'] == 'Late'].groupby('EmployeeID')['Date'].count()

late_days_summary = late_days_summary[late_days_summary.index.isin(low_attendance_employees)]

attendance_rate, low_attendance_employees, late_days_summary
```


