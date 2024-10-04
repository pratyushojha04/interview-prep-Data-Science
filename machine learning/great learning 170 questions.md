## 1. Explain the terms Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL)
- **AI (Artificial Intelligence):** AI is the broader domain focused on creating systems or machines capable of performing tasks that typically require human intelligence. This includes reasoning, learning, problem-solving, perception, and language understanding.
- **ML (Machine Learning):** ML is a subset of AI that involves developing algorithms that allow machines to learn from and make decisions based on data. ML systems improve their performance as they are exposed to more data over time.
- **DL (Deep Learning):** DL is a subset of ML that deals with algorithms inspired by the structure and function of the brain's neural networks. It is particularly useful for large datasets and is employed in areas like image and speech recognition. DL models automatically discover representations needed for feature detection and classification from raw data.

In summary, DL is a subset of ML, and ML is a subset of AI.

## 2. What are the different types of Learning/Training models in ML?
- **Supervised Learning:** The model is trained on labeled data, meaning the input data is paired with the correct output. Examples include:
  - Continuous target variable: Linear Regression, Polynomial Regression.
  - Categorical target variable: Logistic Regression, Naive Bayes, KNN, SVM, Decision Trees, Random Forest, etc.
- **Unsupervised Learning:** The model learns patterns from unlabeled data without guidance. It finds hidden patterns or intrinsic structures in input data. Examples include:
  - Clustering: K-means, Hierarchical Clustering.
  - Dimensionality reduction: PCA (Principal Component Analysis).
- **Reinforcement Learning:** The model learns by interacting with an environment, receiving feedback in the form of rewards or penalties. It’s used in applications like robotics, gaming, and autonomous driving.

## 3. What is the difference between deep learning and machine learning?
- **Data Presentation:** Machine learning algorithms generally require structured data, while deep learning networks work with layers of artificial neural networks, allowing them to learn from unstructured or raw data like images, audio, and text.
- **Feature Engineering:** In ML, feature extraction is typically manual. In DL, feature extraction is automated by the network layers.

## 4. What is the main key difference between supervised and unsupervised machine learning?
- **Supervised Learning:** Requires labeled data to train the model. The model predicts outcomes based on the labeled input.
- **Unsupervised Learning:** Works with unlabeled data. The model tries to identify patterns and relationships within the data.

## 5. How do you select important variables while working on a data set?
- **Correlation Analysis:** Identify and discard correlated variables.
- **P-values from Regression:** Use linear regression to check p-values.
- **Feature Selection Methods:** Forward selection, backward elimination, and stepwise selection.
- **Regularization Techniques:** Use Lasso Regression to select important features.
- **Random Forest:** Use variable importance charts.
- **Information Gain:** Evaluate features based on information gain.

## 6. How can one determine which algorithm to use for a dataset?
- **Data Characteristics:** The choice of algorithm depends on the data. For linear data, use linear regression. For non-linear data, methods like decision trees, SVM, or neural networks may be better.
- **Exploratory Data Analysis (EDA):** Perform EDA to understand data structure and relationships.
- **Purpose:** The choice also depends on the specific task, like classification, regression, clustering, etc.

## 7. How are covariance and correlation different?
- **Covariance:** Measures how two variables change together. Positive covariance indicates that variables increase together, while negative covariance indicates they move inversely.
- **Correlation:** Standardizes covariance, giving a value between -1 and 1, making it easier to interpret relationships between variables.

## 8. State the differences between causality and correlation.
- **Causality:** Indicates that one event causes another. It requires more than just correlation.
- **Correlation:** Indicates a relationship between two variables but doesn’t imply causation.

## 9. How do we apply Machine Learning to Hardware?
- **ML Algorithms in System Verilog:** Implement ML algorithms in System Verilog, a hardware description language, and program them onto an FPGA (Field-Programmable Gate Array).

## 10. Explain One-hot encoding and Label Encoding. How do they affect the dimensionality of the dataset?
- **One-hot Encoding:** Converts categorical variables into binary vectors, increasing the dimensionality of the dataset.
- **Label Encoding:** Converts labels into numeric form without increasing the dataset's dimensionality.

## 11. When does regularization come into play in Machine Learning?
Regularization is used to prevent overfitting or underfitting by penalizing complex models. Techniques like Lasso and Ridge regression add a penalty to the loss function to constrain the coefficients and reduce model complexity.

## 12. What is Bias, Variance, and Bias-Variance Tradeoff?
- **Bias:** Error due to overly simplistic models that can lead to underfitting.
- **Variance:** Error due to models being too complex, leading to overfitting.
- **Tradeoff:** Increasing model complexity reduces bias but increases variance. The goal is to find a balance where both bias and variance are minimized.

## 13. How can we relate standard deviation and variance?
Variance is the average of the squared differences from the mean. Standard deviation is the square root of the variance, providing a measure of spread in the same units as the data.

## 14. A dataset has missing values spread along 1 standard deviation from the mean. How much of the data would remain untouched?
About 68% of data lies within 1 standard deviation of the mean in a normal distribution. Therefore, approximately 32% of the data remains unaffected.

## 15. Is a high variance in data good or bad?
High variance indicates that the data is spread out. While diversity in data can be good, it may also indicate noise, which can complicate model training and lead to overfitting.

## 16. If your dataset has high variance, how would you handle it?
- **Bagging:** Use ensemble methods like bagging to reduce variance by training multiple models on different subsets of the data and averaging their predictions.

## 17. A dataset on utilities fraud detection shows a performance score of 98.5%. Is this a good model?
For imbalanced datasets, accuracy is not a reliable performance metric. Instead, use sensitivity, specificity, or other metrics like F1-score, and consider methods like oversampling, undersampling, or adjusting class weights to improve minority class prediction.

## 18. Explain handling of missing or corrupted values in a dataset.
- **Dropping:** Remove rows or columns with missing values if they are few.
- **Imputation:** Replace missing values with a placeholder like the mean, median, or a specific value using functions like `fillna()` in pandas.

## 19. What is Time Series?
A time series is a sequence of data points collected or recorded at regular time intervals. It's used to track changes over time, often analyzed for trends, seasonal patterns, and forecasting.

## 20. What is a Box-Cox transformation?
A Box-Cox transformation is a power transformation used to stabilize variance and make a dataset more normal distribution-like. It includes a parameter, lambda, that when set to 0, makes it equivalent to a log transformation.

## 21. Difference between Stochastic Gradient Descent (SGD) and Gradient Descent (GD):
- **Gradient Descent:** Uses the entire dataset to calculate the gradient and update the model parameters, making it computationally expensive.
- **Stochastic Gradient Descent:** Uses only one random sample per iteration to update the parameters, making it faster but noisier.

## 22. What is the exploding gradient problem?
The exploding gradient problem occurs when large error gradients accumulate during backpropagation, leading to excessively large weight updates and instability in the neural network. It can cause the model’s weights to become too large, resulting in NaN values.

## 23. Advantages and disadvantages of decision trees:
- **Advantages:**
  - Easy to interpret and visualize.
  - Non-parametric, so robust to outliers.
- **Disadvantages:**
  - Prone to overfitting, especially with noisy data.

## 24. Differences between Random Forest and Gradient Boosting Machines:
- **Random Forests:** Combine the predictions of many independent trees (bagging). They are generally robust and easy to tune.
- **Gradient Boosting Machines:** Build trees sequentially, where each new tree attempts to correct errors made by previous ones. They tend to have better performance but require careful tuning and can overfit if not properly managed.

## 25. What is a confusion matrix and why do you need it?
A confusion matrix is a table used to describe the performance of a classification model by displaying the correct and incorrect predictions in a structured format. It helps identify the types of errors the model is making.

## 26. What’s a Fourier transform?
A Fourier Transform converts a time-domain signal into its constituent frequencies. It's widely used in signal processing, where it breaks down signals into sine and cosine components.

## 27. What is Associative Rule Mining (ARM)?
ARM is a technique in data mining used to discover interesting relationships or patterns (association rules) between variables in large datasets. It’s commonly used in market basket analysis to identify sets of items that frequently co-occur.


## 28. What is root cause analysis?
Root Cause Analysis (RCA) is a method of problem-solving used to identify the underlying causes of a problem. It helps in understanding the reasons behind a particular issue to prevent it from recurring.

## 29. Explain Cross-Validation.
Cross-Validation is a technique for assessing the performance of a machine learning model. It involves dividing the dataset into subsets, training the model on some subsets (training set), and validating it on the remaining subsets (validation set). This process is repeated multiple times, and the results are averaged to get a more accurate measure of model performance.

### Types of Cross-Validation:
- **k-Fold Cross-Validation:** The dataset is divided into `k` subsets. The model is trained on `k-1` subsets and validated on the remaining one. This process is repeated `k` times.
- **Leave-One-Out Cross-Validation (LOOCV):** A special case of k-Fold where `k` is equal to the number of data points in the dataset.

## 30. What are overfitting and underfitting?
- **Overfitting:** Occurs when a model is too complex and captures noise in the training data, performing well on training data but poorly on new data.
- **Underfitting:** Occurs when a model is too simple to capture the underlying patterns in the data, leading to poor performance on both training and new data.

## 31. What is a recommendation system?
A recommendation system is a type of information filtering system that predicts and recommends items to users based on preferences, behavior, or other data. Examples include movie recommendations on Netflix or product recommendations on Amazon.

### Types of Recommendation Systems:
- **Collaborative Filtering:** Recommends items based on the preferences of similar users.
- **Content-Based Filtering:** Recommends items similar to what the user has liked in the past.
- **Hybrid Systems:** Combine collaborative and content-based filtering methods to improve recommendation accuracy.

## 32. What is a normal distribution?
A normal distribution, also known as the Gaussian distribution, is a symmetric, bell-shaped distribution where most of the observations cluster around the central peak. The mean, median, and mode of a normal distribution are all equal.

## 33. What are the feature vectors?
Feature vectors are numeric representations of objects or instances used in machine learning algorithms. Each feature vector contains information (features) that describes a particular instance and is typically represented as a one-dimensional array.

## 34. What are hyperparameters?
Hyperparameters are the external configurations of a machine learning model that are not learned from the data but set before the training process. Examples include learning rate, number of trees in a random forest, and the number of hidden layers in a neural network.

## 35. Explain the difference between classification and regression.
- **Classification:** Involves predicting categorical labels (e.g., spam or not spam).
- **Regression:** Involves predicting continuous numeric values (e.g., predicting house prices).

## 36. Explain SVM (Support Vector Machine) algorithm in detail.
Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates the classes in the feature space.

### Key Points:
- **Support Vectors:** The data points closest to the hyperplane that influence its position.
- **Kernel Trick:** SVM can efficiently perform a non-linear classification using the kernel trick, mapping input features into higher-dimensional spaces.

## 37. Why is “Naive” Bayes naive?
Naive Bayes is termed "naive" because it assumes that all features are independent of each other, which is often not the case in real-world data. Despite this, Naive Bayes performs surprisingly well in many situations.

## 38. Explain the concept of the ROC curve.
The ROC (Receiver Operating Characteristic) curve is a graphical plot used to assess the performance of a binary classification model. It plots the true positive rate (sensitivity) against the false positive rate (1-specificity) at various threshold settings.

### Key Points:
- **AUC (Area Under the Curve):** The area under the ROC curve provides a single measure of the model's discriminatory ability. A model with an AUC of 1 is perfect, while an AUC of 0.5 indicates no discrimination.

## 39. Explain ensemble learning.
Ensemble learning is a technique that combines the predictions of multiple base models to produce a better-performing model. It helps in reducing variance (bagging), bias (boosting), or improving predictions (stacking).

### Examples:
- **Bagging:** Random Forest.
- **Boosting:** Gradient Boosting Machines (GBM), AdaBoost, XGBoost.
- **Stacking:** Combining multiple models through a meta-model.

## 40. What is the curse of dimensionality?
The curse of dimensionality refers to the challenges and inefficiencies that arise when working with data in high-dimensional spaces. As the number of dimensions increases, the volume of the space increases exponentially, making the available data sparse and causing models to struggle with overfitting and computational inefficiency.

## 41. What are dimensionality reduction techniques?
Dimensionality reduction techniques are used to reduce the number of input variables or features in a dataset while retaining as much information as possible.

### Common Techniques:
- **PCA (Principal Component Analysis):** Reduces dimensionality by transforming the data into a set of orthogonal (uncorrelated) components.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Non-linear dimensionality reduction used for visualization.
- **LDA (Linear Discriminant Analysis):** Finds a linear combination of features that separates two or more classes.

## 42. What is data normalization and why do we need it?
Data normalization is the process of scaling individual data points to have a mean of 0 and a standard deviation of 1 (standardization) or to a range of [0, 1]. Normalization is needed to ensure that features contribute equally to the model, especially in algorithms like k-NN, SVM, and neural networks.

## 43. What is exploding gradient?
Exploding gradient is a problem that occurs during the training of deep neural networks where the gradients grow uncontrollably large, causing the model’s weights to become too large, resulting in NaN values or infinite values in the network.

## 44. What is the role of a cost function?
A cost function measures the error or difference between the predicted values and the actual values in a model. It helps in optimizing the model by minimizing the cost function during the training process.

### Common Cost Functions:
- **MSE (Mean Squared Error):** Used for regression tasks.
- **Cross-Entropy Loss:** Used for classification tasks.

## 45. Explain Eigenvectors and Eigenvalues.
- **Eigenvectors:** Non-zero vectors that change only in magnitude, not direction, when a linear transformation is applied to them.
- **Eigenvalues:** Scalars that represent the factor by which the magnitude of the eigenvector is scaled during the transformation.

### Applications:
- Used in Principal Component Analysis (PCA) for dimensionality reduction.

## 46. Explain LSTM and its structure.
LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) that is capable of learning long-term dependencies. It solves the vanishing gradient problem by using special gates to control the flow of information.

### Structure:
- **Input Gate:** Controls the extent to which the new information enters the cell state.
- **Forget Gate:** Controls the extent to which information from the previous cell state should be forgotten.
- **Output Gate:** Controls the extent to which the cell state contributes to the output.

## 47. What is the difference between Type I and Type II errors?
- **Type I Error:** Also known as a false positive, it occurs when the null hypothesis is incorrectly rejected.
- **Type II Error:** Also known as a false negative, it occurs when the null hypothesis is incorrectly accepted.

## 48. Explain the bias-variance tradeoff.
The bias-variance tradeoff is the balance between two types of errors:
- **Bias Error:** Error due to overly simplistic models that fail to capture the underlying data patterns (underfitting).
- **Variance Error:** Error due to overly complex models that capture noise as well as data patterns (overfitting).
The goal is to find a model with low bias and low variance.

## 49. What is an F1 Score?
The F1 Score is a measure of a model’s accuracy that considers both precision and recall. It is the harmonic mean of precision and recall, providing a single metric for performance, especially useful for imbalanced datasets.

### Formula:
\[ F1 = 2 \times \left(\frac{Precision \times Recall}{Precision + Recall}\right) \]

## 50. What is A/B testing?
A/B testing is a statistical method used to compare two versions of a webpage or app feature to determine which one performs better. It involves dividing users into two groups: one experiences version A, and the other experiences version B, with the aim of identifying which version leads to better outcomes (e.g., higher conversion rates).
# NLP Interview Questions

NLP or Natural Language Processing helps machines analyze natural languages with the intention of learning them. It extracts information from data by applying machine learning algorithms. Apart from learning the basics of NLP, it is important to prepare specifically for the interviews. Check out the top NLP Interview Questions.

### 48. How does the SVM algorithm deal with self-learning?
SVM has a learning rate and expansion rate which takes care of this. The learning rate compensates or penalizes the hyperplanes for making all the wrong moves, and the expansion rate deals with finding the maximum separation area between classes.

### 49. What are Kernels in SVM? List popular kernels used in SVM along with a scenario of their applications.
The function of the kernel is to take data as input and transform it into the required form. A few popular Kernels used in SVM are as follows: RBF, Linear, Sigmoid, Polynomial, Hyperbolic, Laplace, etc.

### 50. What is Kernel Trick in an SVM Algorithm?
Kernel Trick is a mathematical function that, when applied to data points, can find the region of classification between two different classes. Based on the choice of function, be it linear or radial, which purely depends upon the distribution of data, one can build a classifier.

### 51. What are ensemble models? Explain how ensemble techniques yield better learning as compared to traditional classification ML algorithms.
An ensemble is a group of models that are used together for prediction, both in classification and regression tasks. Ensemble learning helps improve ML results because it combines several models. By doing so, it allows for a better predictive performance compared to a single model. They are superior to individual models as they reduce variance, average out biases, and have lesser chances of overfitting.

### 52. What are overfitting and underfitting? Why does the decision tree algorithm often suffer from overfitting problems?
- **Overfitting** is a statistical model or machine learning algorithm that captures the data’s noise.
- **Underfitting** is a model or machine learning algorithm that does not fit the data well enough and occurs if the model or algorithm shows low variance but high bias.

In decision trees, overfitting occurs when the tree is designed to fit all samples in the training dataset perfectly. This results in branches with strict rules or sparse data and affects the accuracy when predicting samples that aren’t part of the training set.

### 53. What is OOB error and how does it occur?
For each bootstrap sample, there is one-third of the data that was not used in the creation of the tree, i.e., it was out of the sample. This data is referred to as out-of-bag data. In order to get an unbiased measure of the accuracy of the model over test data, out-of-bag error is used. The out-of-bag data is passed for each tree, and the outputs are aggregated to give out-of-bag error. This percentage error is quite effective in estimating the error in the testing set and does not require further cross-validation.

### 54. Why is boosting a more stable algorithm compared to other ensemble algorithms?
Boosting focuses on errors found in previous iterations until they become obsolete. Whereas in bagging, there is no corrective loop. This is why boosting is a more stable algorithm compared to other ensemble algorithms.

### 55. How do you handle outliers in the data?
Outliers are observations in the dataset that are far away from other observations in the dataset. We can discover outliers using tools and functions like box plots, scatter plots, Z-Score, IQR score, etc., and then handle them based on the visualization we have got. To handle outliers, we can cap at some threshold, use transformations to reduce skewness of the data, and remove outliers if they are anomalies or errors.

### 56. List popular cross-validation techniques.
There are mainly six types of cross-validation techniques. They are as follows:
- K-fold
- Stratified k-fold
- Leave-one-out
- Bootstrapping
- Random search cv
- Grid search cv

### 57. Is it possible to test for the probability of improving model accuracy without cross-validation techniques? If yes, please explain.
Yes, it is possible to test for the probability of improving model accuracy without cross-validation techniques. We can do so by running the ML model for, say, n number of iterations and recording the accuracy. Plot all the accuracies and remove the 5% of low probability values. Measure the left (low) cutoff and right (high) cutoff. With the remaining 95% confidence, we can say that the model can go as low or as high as mentioned within cutoff points.

### 58. Name a popular dimensionality reduction algorithm.
Popular dimensionality reduction algorithms are Principal Component Analysis and Factor Analysis. Principal Component Analysis creates one or more index variables from a larger set of measured variables. Factor Analysis is a model of the measurement of a latent variable. This latent variable cannot be measured with a single variable and is seen through a relationship it causes in a set of y variables.

### 59. How can we use a dataset without the target variable in supervised learning algorithms?
Input the dataset into a clustering algorithm, generate optimal clusters, and label the cluster numbers as the new target variable. Now, the dataset has independent and target variables present. This ensures that the dataset is ready to be used in supervised learning algorithms.

### 60. List all types of popular recommendation systems. Name and explain two personalized recommendation systems along with their ease of implementation.
- Popularity-based recommendation
- Content-based recommendation
- User-based collaborative filter
- Item-based recommendation

Personalized Recommendation systems are:
- Content-based recommendations
- User-based collaborative filter
- Item-based recommendations

User-based collaborative filter and item-based recommendations are more personalized. Easy to maintain: Similarity matrix can be maintained easily with Item-based recommendations.

### 61. How do we deal with sparsity issues in recommendation systems? How do we measure its effectiveness? Explain.
Singular value decomposition can be used to generate the prediction matrix. RMSE is the measure that helps us understand how close the prediction matrix is to the original matrix.

### 62. Name and define techniques used to find similarities in the recommendation system.
Pearson correlation and Cosine correlation are techniques used to find similarities in recommendation systems.

### 63. State the limitations of Fixed Basis Function.
Linear separability in feature space doesn’t imply linear separability in input space. So, Inputs are non-linearly transformed using vectors of basic functions with increased dimensionality. Limitations of Fixed basis functions are:
- Non-Linear transformations cannot remove overlap between two classes but they can increase overlap.
- Often it is not clear which basis functions are the best fit for a given task. So, learning the basic functions can be useful over using fixed basis functions.
- If we want to use only fixed ones, we can use a lot of them and let the model figure out the best fit but that would lead to overfitting the model, thereby making it unstable.

### 64. Define and explain the concept of Inductive Bias with some examples.
Inductive Bias is a set of assumptions that humans use to predict outputs given inputs that the learning algorithm has not encountered yet. When we are trying to learn Y from X and the hypothesis space for Y is infinite, we need to reduce the scope by our beliefs/assumptions about the hypothesis space, which is also called inductive bias. Through these assumptions, we constrain our hypothesis space and also get the capability to incrementally test and improve on the data using hyper-parameters. Examples:
- We assume that Y varies linearly with X while applying Linear regression.
- We assume that there exists a hyperplane separating negative and positive examples.

### 65. Explain the term instance-based learning.
Instance-Based Learning is a set of procedures for regression and classification that produce a class label prediction based on resemblance to its nearest neighbors in the training dataset. These algorithms just collect all the data and get an answer when required or queried. In simple words, they are a set of procedures for solving new problems based on the solutions of already solved problems in the past which are similar to the current problem.

### 66. Keeping train and test split criteria in mind, is it good to perform scaling before the split or after the split?
Scaling should be done post-train and test split ideally. If the data is closely packed, then scaling post or pre-split should not make much difference.

### 67. Define precision, recall, and F1 Score.
- **True Positives (TP)** – These are the correctly predicted positive values. It implies that the value of the actual class is yes and the value of the predicted class is also yes.
- **True Negatives (TN)** – These are the correctly predicted negative values. It implies that the value of the actual class is no and the value of the predicted class is also no.
- **False positives and false negatives** – These values occur when your actual class contradicts with the predicted class.

Now,
- **Recall**, also known as Sensitivity, is the ratio of true positive rate (TP), to all observations in the actual class – yes.
  - `Recall = TP/(TP+FN)`

- **Precision** is the ratio of positive predictive value, which measures the amount of accurate positives the model predicted vis-à-vis the number of positives it claims.
  - `Precision = TP/(TP+FP)`

- **Accuracy** is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations.
  - `Accuracy = (TP+TN)/(TP+FP+FN+TN)`

- **F1 Score** is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution.
  - `F1 Score = 2*(Recall * Precision) / (Recall + Precision)`

### 68. List the methods for reducing dimensionality.
Some methods for reducing dimensionality are:
- Factor analysis
- Independent component analysis
- Missing values ratio
- Low variance filter
- High correlation filter
- Random forest

### 69. List feature selection methods.
- The feature selection methods are:
- Information gain
- Chi-Square test
- Anova
- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Trees
- Random Forest
- Recursive Feature Elimination (RFE)

### 70. Explain the box cox method.
The Box-Cox transformation transforms our data so that it closely resembles a normal distribution. If we have a variable that is not normally distributed, we can perform this transformation to make it normal or nearly normal. 
# Machine Learning Interview Questions and Answers

### 71. Explain how a Naive Bayes Classifier works.
Naive Bayes classifiers are a family of algorithms derived from the Bayes theorem of probability. It works on the fundamental assumption that every set of two features being classified is independent of each other, and every feature makes an equal and independent contribution to the outcome.

**Example:** If you're classifying emails as spam or not spam, Naive Bayes assumes that the occurrence of a word in an email is independent of the occurrence of any other word.

### 72. What do the terms prior probability and marginal likelihood in the context of Naive Bayes theorem mean?
- **Prior probability** is the percentage of dependent binary variables in the dataset. For example, if 65% of emails in a dataset are spam and 35% are not, then the prior probability of an email being spam is 65%.
- **Marginal likelihood** is the denominator of the Bayes equation. It ensures that the posterior probability is valid by making its total area equal to 1.

### 73. Explain the difference between Lasso and Ridge.
- **Lasso (L1) and Ridge (L2)** are regularization techniques where coefficients are penalized to find the optimum solution.
  - In **Ridge**, the penalty is defined by the sum of the squares of the coefficients.
  - In **Lasso**, the penalty is the sum of the absolute values of the coefficients.
  - **ElasticNet** is a hybrid penalizing function that combines Lasso and Ridge.

### 74. What’s the difference between probability and likelihood?
- **Probability** measures the likelihood that a specific event will occur. It attaches to possible outcomes.
- **Likelihood** is a function of parameters within the parameter space that describes the probability of obtaining the observed data. It attaches to hypotheses.

### 75. Why would you prune your tree?
Pruning reduces the size of a decision tree by removing branches that are redundant, thus minimizing the chances of overfitting. This process turns branches of a decision tree into leaf nodes and removes unnecessary leaf nodes from the original branch.

### 76. Model accuracy or model performance? Which one will you prefer and why?
- **Model performance** can mean different things depending on context:
  - If it refers to speed, then for real-time applications, speed is crucial.
  - If it refers to metrics, for imbalanced datasets, metrics like the F1 score, precision, and recall may be more important than accuracy.

### 77. List the advantages and limitations of the Temporal Difference Learning Method.
**Advantages:**
- Learns in every step (online/offline).
- Can learn from incomplete sequences.
- Works in continuous environments.
- Lower variance and more efficient compared to the Monte Carlo method.

**Limitations:**
- Biased estimation.
- Sensitive to initialization.

### 78. How would you handle an imbalanced dataset?
**Sampling Techniques:**
- **Under Sampling:** Reducing the size of the majority class to match the minority class.
- **Over Sampling:** Increasing the minority class by creating synthetic samples.

**Other Techniques:**
- **Cluster-Based Over Sampling:** K-means clustering is applied to minority and majority class instances, and each cluster is oversampled.
- **SMOTE:** Creates synthetic instances for the minority class to balance the dataset.

### 79. Mention some of the EDA Techniques.
**Exploratory Data Analysis (EDA) Techniques:**
- **Visualization:**
  - Univariate, Bivariate, Multivariate visualization.
- **Missing Value Treatment:** Replace missing values with mean/median.
- **Outlier Detection:** Identify outliers using Boxplot and apply IQR to set boundaries.
- **Transformation:** Apply transformations based on feature distribution.
- **Scaling the Dataset:** Apply MinMax, Standard Scaler, or Z Score Scaling.
- **Feature Engineering:** Create derivative fields to enhance model performance.
- **Dimensionality Reduction:** Reduce data volume without losing significant information.

### 80. Mention why feature engineering is important in model building and list out some of the techniques used for feature engineering.
**Importance:**
- Algorithms require features with specific characteristics to function effectively. Feature engineering prepares and enhances input data for compatibility and performance.

**Techniques:**
- Imputation, Binning, Outliers Handling, Log Transform, Grouping Operations, One-Hot Encoding, Feature Split, Scaling, and Extracting Date.

### 81. Differentiate between Statistical Modeling and Machine Learning.
- **Machine Learning models** are focused on making accurate predictions.
- **Statistical models** are designed to infer relationships between variables.

### 82. Differentiate between Boosting and Bagging.
**Bagging:**
- Reduces variance in high-variance algorithms like decision trees.
- Uses multiple models trained on different samples and averages their results.

**Boosting:**
- Sequentially applies weak classifiers, each compensating for the weaknesses of the previous ones.
- Algorithms include AdaBoost, Gradient Boosting, and XGBoost.

### 83. What is the significance of Gamma and Regularization in SVM?
- **Gamma** determines the influence of individual data points.
- **Regularization** (lambda) controls the trade-off between fitting the training data and generalizing to unseen data.

### 84. Define ROC curve.
The ROC curve is a graphical representation of the trade-off between the true positive rate and the false positive rate at various thresholds. It helps evaluate model performance.

### 85. What is the difference between a generative and discriminative model?
- **Generative models** learn the distribution of each class.
- **Discriminative models** learn the boundary between classes and are generally better for classification tasks.

### 86. What are hyperparameters and how are they different from parameters?
- **Parameters** are internal to the model and estimated from the training data (e.g., weights, biases).
- **Hyperparameters** are external to the model and are set before training (e.g., learning rate, hidden layers).

### 87. What is shattering a set of points? Explain VC dimension.
- **Shattering** occurs when a classifier can perfectly separate a given configuration of points for all possible assignments.
- **VC dimension** is the cardinality of the largest set of points that can be shattered by the classifier. It helps choose the appropriate classifier based on data type.

### 88. What are some differences between a linked list and an array?
**Array:**
- Elements are well-indexed.
- Operations are faster.
- Fixed size.
- Memory is assigned during compile time.
- Elements are stored consecutively.

**Linked List:**
- Elements need cumulative access.
- Operations take linear time.
- Dynamic and flexible.
- Memory is allocated during runtime.
- Elements are stored randomly.
- Efficient memory utilization.

### 89. What is the `meshgrid()` method and the `contourf()` method? State some uses of both.
- **`meshgrid()`**: Creates a grid using 1-D arrays of x and y-axis inputs.
- **`contourf()`**: Draws filled contours using the given inputs. It is used after creating a grid with `meshgrid()`.

### 90. Describe a hash table.
Hashing identifies unique objects from a group by converting large keys into small keys. These values are stored in a data structure known as a hash table.
# Machine Learning and Data Science Interview Questions

## 91. Advantages and Disadvantages of Neural Networks

### Advantages
- **Information Storage**: Neural networks store information across the network rather than in a database.
- **Accuracy with Inadequate Information**: They can perform well even with incomplete or noisy data.
- **Parallel Processing**: They leverage parallel processing capabilities and distributed memory.

### Disadvantages
- **Hardware Requirements**: They require processors capable of parallel processing.
- **Opacity**: Their functioning can be opaque, which reduces trust, especially when diagnosing issues.
- **Training Duration**: The training duration is often unknown, and convergence is monitored via error values, which might not always provide optimal results.

## 92. Training a Large Dataset with Limited RAM

To handle a 12GB dataset on a machine with 3GB RAM:
- **Use NumPy Arrays**: Load the data into arrays that can be mapped without complete loading into memory.
- **Batch Processing**: Divide the data into batches to process in chunks, ensuring manageable memory usage.

## 93. Binarize Data

Convert data into binary based on a threshold. Values below the threshold are set to 0, and those above are set to 1.

```python
from sklearn.preprocessing import Binarizer
import pandas as pd
import numpy as np

names_list = ['Alaska', 'Pratyush', 'Pierce', 'Sandra', 'Soundarya', 'Meredith', 'Richard', 'Jackson', 'Tom', 'Joe']
data_frame = pd.read_csv('url', names=names_list)
array = data_frame.values

# Splitting the array into input and output 
A = array[:7]
binarizer = Binarizer(threshold=0.0).fit(A)
binaryA = binarizer.transform(A)

np.set_printoptions(precision=5)
print(binaryA[:7])
```



### 94. What is an Array?

An array is a collection of similar items stored in contiguous memory locations. Each element consumes a specific amount of memory based on its data type. For example, an `int` typically uses 4 bytes, while a `char` uses 1 byte.

**Example:**

```python
fruits = ['apple', 'banana', 'pineapple']
```


### 95. What are the advantages and disadvantages of using an Array?
### Advantages:

1.Random access is enabled.

2.Saves memory.

3.Cache-friendly.

4.Predictable compile timing.

5.Helps in code reusability.

### Disadvantages:

Adding and deleting elements is time-consuming as elements need to be reordered.
Requires contiguous blocks of memory, which may cause overhead if large blocks are not available.



### 101. Explain Eigenvectors and Eigenvalues

Linear transformations are helpful to understand using eigenvectors. They find their prime usage in the creation of covariance and correlation matrices in data science.

Simply put, eigenvectors are directional entities along which linear transformation features like compression, flip, etc., can be applied.

Eigenvalues are the magnitude of the linear transformation features along each direction of an eigenvector.

### 102. How would you define the number of clusters in a clustering algorithm?

The number of clusters can be determined by finding the silhouette score. Often we aim to get some inferences from data using clustering techniques so that we can have a broader picture of the number of classes being represented by the data. In this case, the silhouette score helps us determine the number of cluster centers to cluster our data along.

Another technique that can be used is the elbow method.

### 103. What are the performance metrics that can be used to estimate the efficiency of a linear regression model?

The performance metrics used are:
- Mean Squared Error (MSE)
- R² Score
- Adjusted R² Score
- Mean Absolute Error (MAE)

### 104. What is the default method of splitting in decision trees?

The default method of splitting in decision trees is the Gini Index. The Gini Index measures the impurity of a particular node.

This can be changed by adjusting classifier parameters.

### 105. How is p-value useful?

The p-value gives the probability of the null hypothesis being true. It provides statistical significance for the results. In other words, the p-value determines the confidence of a model in a particular output.

### 106. Can logistic regression be used for classes more than 2?

No, logistic regression is inherently a binary classifier. For multi-class classification, algorithms like Decision Trees or Naïve Bayes' Classifiers are more appropriate.

### 107. What are the hyperparameters of a logistic regression model?

Hyperparameters of a Logistic Regression model include:
- Classifier penalty
- Classifier solver
- Classifier C

These can be specified with values in Grid Search for hyperparameter tuning.

### 108. Name a few hyper-parameters of decision trees.

Important hyperparameters of decision trees include:
- Splitting criteria
- Min_samples_split
- Min_samples_leaf
- Max_depth

### 109. How to deal with multicollinearity?

Multicollinearity can be dealt with by:
- Removing highly correlated predictors from the model.
- Using Partial Least Squares Regression (PLS) or Principal Components Analysis (PCA).

### 110. What is Heteroscedasticity?

Heteroscedasticity is a situation where the variance of a variable is unequal across the range of values of the predictor variable. It should be avoided in regression as it introduces unnecessary variance.

### 111. Is ARIMA model a good fit for every time series problem?

No, the ARIMA model is not suitable for every type of time series problem. There are situations where the ARMA model and others are also useful. ARIMA is best when capturing different standard temporal structures in time series data.

### 112. How do you deal with the class imbalance in a classification problem?

Class imbalance can be addressed by:
- Using class weights
- Using sampling techniques
- Using SMOTE (Synthetic Minority Over-sampling Technique)
- Choosing loss functions like Focal Loss

### 113. What is the role of cross-validation?

Cross-validation is a technique used to improve the performance of a machine learning algorithm. It involves splitting the dataset into smaller parts, using a random part as the test set, and all other parts as training sets. This helps in assessing the model’s performance more reliably.

### 114. What is a voting model?

A voting model is an ensemble model that combines several classifiers. For a classification-based model, it aggregates the classifications from all models and selects the most voted option from all the given classes in the target column.

### 115. How to deal with very few data samples? Is it possible to make a model out of it?

With very few data samples, you can use oversampling to generate new data points. This helps in creating a more robust model.

### 116. What are the hyperparameters of an SVM?

Hyperparameters of an SVM model include:
- Gamma value
- C value
- Type of kernel

### 117. What is Pandas Profiling?

Pandas profiling is a tool that provides statistics on NULL values and usable values in a dataset. It aids in variable selection and data preprocessing for building models.

### 118. What impact does correlation have on PCA?

If data is correlated, PCA does not work well because the effective variance of variables decreases. Hence, correlated data can lead to less effective PCA results.

### 119. How is PCA different from LDA?

- PCA (Principal Component Analysis) is unsupervised and focuses on variance.
- LDA (Linear Discriminant Analysis) is supervised and considers the distribution of classes.

### 120. What distance metrics can be used in KNN?

Distance metrics used in KNN include:
- Manhattan
- Minkowski
- Tanimoto
- Jaccard
- Mahalanobis

### 121. Which metrics can be used to measure correlation of categorical data?

The Chi-square test can be used to measure correlation between categorical predictors.

### 122. Which algorithm can be used in value imputation in both categorical and continuous categories of data?

KNN (K-Nearest Neighbors) can be used for imputation of both categorical and continuous variables.

### 123. When should ridge regression be preferred over lasso?

Ridge regression should be used when you want to retain all predictors and only reduce their coefficient values, rather than eliminating any predictors.

### 124. Which algorithms can be used for important variable selection?

Algorithms for variable selection include:
- Random Forest
- XGBoost
- Variable importance plots

### 125. What ensemble technique is used by Random Forests?

Random Forests use the bagging technique. They are a collection of decision trees that work on sampled data from the original dataset, with the final prediction being the average of all tree predictions.



### 126. What ensemble technique is used by gradient boosting trees?

Boosting is the technique used by Gradient Boosting Machines (GBM).

### 127. If we have a high bias error what does it mean? How to treat it?

High bias error means that the model is ignoring important trends in the data and is underfitting.

To reduce underfitting:
- Increase the complexity of the model.
- Add more features to the model.
- Remove noise from the data to ensure that the model can capture the most important signals.
- Increase the number of epochs to extend the training duration of the model, which may help in reducing the error.

### 128. Which type of sampling is better for a classification model and why?

Stratified sampling is better for classification problems because it maintains the balance of classes in both the training and test sets. This ensures that the proportion of classes is preserved, leading to better model performance. Random sampling may result in imbalanced classes in train and test sets, which can degrade the model's performance.

### 129. What is a good metric for measuring the level of multicollinearity?

Variance Inflation Factor (VIF) or 1/tolerance is a good metric for measuring multicollinearity. VIF quantifies how much the variance of a predictor is inflated due to multicollinearity with other predictors. 

A rule of thumb for interpreting VIF:
- 1 = Not correlated
- Between 1 and 5 = Moderately correlated
- Greater than 5 = Highly correlated

### 130. When can a categorical value be treated as a continuous variable and what effect does it have when done so?

A categorical predictor can be treated as a continuous variable when the data points represent ordinal data. If the predictor variable is ordinal, treating it as continuous can improve model performance by capturing the ordinal relationship in the data.

### 131. What is the role of maximum likelihood in logistic regression?

Maximum likelihood estimation helps in estimating the most probable values of the predictor variable coefficients. It produces results that are most likely or closest to the true values.

### 132. Which distance do we measure in the case of KNN?

In KNN, the distance metrics commonly used include:
- Hamming distance for categorical data
- Euclidean distance for continuous data (also used in K-means)

### 133. What is a pipeline?

A pipeline is a systematic way of structuring a machine learning workflow such that each step of the process (e.g., preprocessing, modeling) is performed in sequence. It enables serialization of tasks and execution on multiple threads using composite estimators in libraries like scikit-learn.

### 134. Which sampling technique is most suitable when working with time-series data?

A custom iterative sampling approach is most suitable for time-series data. This method involves continuously adding samples to the train set while ensuring that validation samples are kept separate and used appropriately.

### 135. What are the benefits of pruning?

Pruning helps by:
- Reducing overfitting
- Shortening the size of the decision tree
- Reducing model complexity
- Increasing bias slightly

### 136. What is normal distribution?

Normal distribution has the following properties:
- The mean, mode, and median are all equal.
- The curve is symmetric around the mean (μ).
- Exactly half of the values lie to the left of the center, and exactly half lie to the right.
- The total area under the curve is 1.

### 137. What is the 68 percent rule in normal distribution?

In a normal distribution, approximately 68% of the data points lie within one standard deviation of the mean. This rule reflects the bell-shaped nature of the normal distribution.

### 138. What is a chi-square test?

A chi-square test assesses whether a sample data matches the population. It compares two variables in a contingency table to determine if they are related. A small chi-square statistic indicates that observed data fits the expected data well.

### 139. What is a random variable?

A random variable is a set of possible values from a random experiment. For example:
- Tossing a coin can result in Heads or Tails.
- Rolling a dice can result in one of six values.

### 140. What is the degree of freedom?

Degrees of freedom represent the number of independent values or quantities that can be assigned to a statistical distribution. It is used in hypothesis testing and chi-square tests.

### 141. Which kind of recommendation system is used by Amazon to recommend similar items?

Amazon uses a collaborative filtering algorithm for recommending similar items. This approach maps user similarities based on their buying patterns and preferences.

### 142. What is a false positive?

A false positive is a test result that incorrectly indicates the presence of a condition or attribute. 

Example: Stress testing may show a significant number of false positives in detecting heart disease in women.

### 143. What is a false negative?

A false negative is a test result that incorrectly indicates the absence of a condition or attribute. 

Example: A pregnancy test might indicate a negative result when the person is actually pregnant.

### 144. What is the error term composed of in regression?

In regression, the error term is composed of:
- Bias error
- Variance error
- Irreducible error

Bias and variance errors can be reduced, but irreducible error is inherent and cannot be eliminated.

### 145. Which performance metric is better, R² or adjusted R²?

Adjusted R² is generally better because it accounts for the number of predictors and adjusts for model complexity. R² can be misleading as it increases with the addition of more predictors, regardless of their relevance.

### 146. What’s the difference between Type I and Type II error?

- Type I error (False Positive): Rejecting a null hypothesis that is actually true.
- Type II error (False Negative): Failing to reject a null hypothesis that is actually false.

### 147. What do you understand by L1 and L2 regularization?

- **L1 Regularization**: Encourages sparsity by setting many weights to zero. It corresponds to a Laplace prior.
- **L2 Regularization**: Spreads the error among all terms, discouraging large coefficients. It corresponds to a Gaussian prior.

### 148. Which one is better, Naive Bayes Algorithm or Decision Trees?

The choice depends on the problem, but here are some general advantages:
- **Naive Bayes**:
  - Works well with small datasets.
  - Less prone to overfitting.
  - Smaller and faster in processing.
- **Decision Trees**:
  - Flexible, easy to understand, and debug.
  - Requires no preprocessing or feature transformation.
  - Prone to overfitting but can be mitigated using pruning or Random Forests.

### 149. What do you mean by the ROC curve?

The ROC (Receiver Operating Characteristic) curve illustrates the diagnostic ability of a binary classifier by plotting True Positive Rate against False Positive Rate at various threshold settings. The performance metric for ROC is AUC (Area Under Curve), with a higher AUC indicating better model performance.

### 150. What do you mean by AUC curve?

AUC (Area Under Curve) measures the overall performance of a classifier. A higher AUC value indicates better prediction power of the model.

### 151. What is log likelihood in logistic regression?

Log likelihood is the sum of the natural logs of the likelihoods for each record. It is used to estimate the predictive power of the model. Deviance and likelihood values help compare different models, with the accuracy of the model always being evaluated on unseen data.


### 152. How would you evaluate a logistic regression model?

1. **Akaike Information Criterion (AIC)**: This criterion helps assess the quality of a model relative to others. It estimates the amount of information lost by the model. Models with lower AIC values are preferred as they indicate less information loss.

2. **Receiver Operating Characteristic (ROC) Curve**: This curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. The performance metric for the ROC curve is the Area Under the Curve (AUC). A higher AUC indicates better model performance.

3. **Confusion Matrix**: This is a table showing the counts of actual versus predicted classifications, which helps determine accuracy, precision, recall, and F1-score.

### 153. What are the advantages of SVM algorithms?

1. **Better Performance**: SVMs often provide better practical performance compared to other classifiers.
2. **Computational Efficiency**: SVMs are generally computationally cheaper (O(N²K) where K is the number of support vectors) compared to logistic regression (O(N³)).
3. **Dependence on Subset of Points**: SVMs focus on a subset of points (support vectors) that define the decision boundary, rather than the entire dataset, making them efficient in high-dimensional spaces.

### 154. Why does XGBoost perform better than SVM?

1. **Ensemble Method**: XGBoost combines multiple weak learners (trees) to create a robust model, leveraging boosting to improve accuracy.
2. **Kernel Limitations**: SVMs need kernels to handle non-linearly separable data, and selecting the right kernel can be challenging. XGBoost avoids this by using decision trees directly.

### 155. What is the difference between SVM Rank and SVR (Support Vector Regression)?

- **SVM Rank**: Used for ranking tasks, where the goal is to order items rather than predict their exact values.
- **SVR**: Used for regression tasks, predicting continuous values where the model fits a function to the data.

### 156. What is the difference between the normal soft margin SVM and SVM with a linear kernel?

- **Hard-Margin SVM**: Assumes data is perfectly separable without errors.
- **Soft-Margin SVM**: Allows some errors (soft margin) to handle non-linearly separable data, balancing between maximizing margin and minimizing classification error.
- **Kernel Trick**: Applies to SVMs when data isn't linearly separable. It maps data into a higher-dimensional space where a linear separator might be feasible.

### 157. How is a linear classifier relevant to SVM?

An SVM is a type of linear classifier when not using a kernel. It learns a linear decision boundary to separate classes in the feature space.

### 158. What are the advantages of using a Naive Bayes classifier?

1. **Simplicity**: Very simple, easy to implement, and fast.
2. **Convergence**: If the conditional independence assumption holds, it converges quicker than discriminative models like logistic regression.
3. **Practical Performance**: Works well even if the Naive Bayes assumption doesn’t hold.
4. **Less Training Data Required**: Can work with smaller datasets.
5. **Scalability**: Scales linearly with the number of predictors and data points.
6. **Versatility**: Can be used for both binary and multiclass classification problems.
7. **Probabilistic Predictions**: Provides probabilistic predictions.
8. **Handles Various Data Types**: Suitable for both continuous and discrete data.
9. **Robustness**: Not sensitive to irrelevant features.

### 159. Are Gaussian Naive Bayes and Binomial Naive Bayes the same?

- **Binomial Naive Bayes**: Assumes features are binary (0s and 1s). For example, in text classification, 0 might indicate "word does not occur" and 1 might indicate "word occurs."

- **Gaussian Naive Bayes**: Assumes features follow a normal distribution. Used when features are continuous, such as in the Iris dataset where features like sepal width and petal length can have a range of values.

### 160. What is the difference between the Naive Bayes Classifier and the Bayes classifier?

- **Naive Bayes Classifier**: Assumes conditional independence between features given the class label. 
  - Formula: \( P(X|Y, Z) = P(X|Z) \)

- **Bayes Classifier**: More general Bayesian networks allow for modeling conditional dependencies between features. Bayesian networks use scoring functions to determine the structure and parameters of the model, focusing on the Markov blanket of nodes.

### 161. In what real-world applications is the Naive Bayes classifier used?

1. **Spam Detection**: Classifying emails as spam or not spam.
2. **News Classification**: Categorizing news articles into topics like technology, politics, or sports.
3. **Sentiment Analysis**: Determining if a piece of text expresses positive or negative emotions.
4. **Face Recognition**: Used in face recognition software.

### 162. Is Naive Bayes supervised or unsupervised?

Naive Bayes is a **supervised learning** algorithm. It uses labeled data to learn the probability distributions of the features and make predictions.

### 163. What do you understand by selection bias in Machine Learning?

Selection bias occurs when the sample used for analysis is not representative of the population intended to be analyzed, often due to non-random selection methods. This bias can lead to inaccurate conclusions. Types of selection bias include:

- **Sampling Bias**: Systematic error due to non-random sampling.
- **Time Interval Bias**: Results affected by early termination of trials.
- **Data Bias**: Selective use of subsets of data.
- **Attrition Bias**: Loss of participants leading to biased results.

### 164. What do you understand by Precision and Recall?

- **Precision**: The fraction of relevant instances among the retrieved instances. Formula: \( \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \)
- **Recall**: The fraction of relevant instances that were actually retrieved. Formula: \( \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \)

### 165. What Are the Three Stages of Building a Model in Machine Learning?

1. **Understand the Business Problem**: Define the problem and objectives.
2. **Data Preparation**: Acquire, clean, and preprocess data.
3. **Model Building**: Apply machine learning algorithms, train the model, and evaluate its performance.

### 166. How Do You Design an Email Spam Filter in Machine Learning?

1. **Understand the Business Model**: Identify attributes relevant to spam detection.
2. **Data Acquisition**: Collect a dataset of emails classified as spam or not spam.
3. **Data Cleaning**: Preprocess and clean the data.
4. **Exploratory Data Analysis**: Analyze data for patterns and trends.
5. **Model Building**: Use machine learning algorithms like Naive Bayes to create the spam filter.
6. **Model Evaluation**: Test the model with unseen data to check its accuracy.

### 167. What is the difference between Entropy and Information Gain?

- **Entropy**: Measures the uncertainty or disorder in a dataset. Formula: \( H(D) = -\sum_{i} p_i \log(p_i) \)
- **Information Gain**: The reduction in entropy after splitting the dataset based on an attribute. Formula: \( IG(D, A) = H(D) - \sum_{v} \frac{|D_v|}{|D|} H(D_v) \)

### 168. What are collinearity and multicollinearity?

- **Collinearity**: A linear relationship between two predictors.
- **Multicollinearity**: A situation where two or more predictors are highly linearly related, potentially causing issues in regression models.

### 169. What is Kernel SVM?

Kernel SVM is an extension of SVM that uses kernel functions to map data into a higher-dimensional space where a linear separator might be feasible. This allows SVM to handle non-linearly separable data.

### 170. What is the process of carrying out linear regression?

1. **Analyze Correlation**: Examine the relationship between variables.
2. **Estimate the Model**: Fit a linear model to the data.
3. **Evaluate Validity**: Assess the model’s performance and usefulness.

---

**KickStart your Artificial Intelligence Journey with Great Learning** which offers high-rated Artificial Intelligence courses with world-class training by industry leaders. Whether you’re interested in machine learning, data mining, or data analysis, Great Learning has a course for you!
