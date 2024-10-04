### **Introduction to Neural Networks**

A **neural network** is a computational model inspired by the way biological neural networks in the human brain process information. They consist of interconnected units called neurons that work together to solve complex problems, such as image recognition, language processing, and decision-making.

- **Biological Inspiration**: The concept of neural networks is inspired by how neurons in the human brain interact with each other.
- **Artificial Neurons**: An artificial neuron (also called a node) takes in multiple inputs, applies a weight to each, adds a bias term, and passes the result through an activation function.
- **Learning Process**: Neural networks are trained by adjusting weights to minimize the error in the output predictions compared to the actual values, using optimization techniques like gradient descent.

### **Perceptron and Activation Functions**

#### **Perceptron**
A **perceptron** is the simplest type of artificial neural network. It is a binary classifier that can classify input data into one of two categories.

- **Single Layer**: A perceptron consists of a single layer of input neurons connected to an output neuron.
- **Mathematics**: The output \( y \) of a perceptron is given by:

  \[
  y = f(\sum (w_i \cdot x_i) + b)
  \]

  where \( w_i \) represents the weight associated with the \( i \)-th input \( x_i \), and \( b \) is the bias term.
- **Activation Function**: The perceptron uses an activation function to determine whether the output should be activated (e.g., \( 0 \) or \( 1 \)). The common activation function for a perceptron is the **step function**, which outputs 1 if the input is above a threshold and 0 otherwise.

#### **Activation Functions**
An **activation function** introduces non-linearity to a neural network, allowing it to learn complex patterns.

- **Linear Activation Function**: Outputs a linear transformation of inputs. Not suitable for learning complex functions, as it lacks non-linearity.
- **Step Function**: Outputs a fixed value based on a threshold. Used in simple binary classifiers like perceptrons.
- **Sigmoid Function**: Maps inputs to a value between 0 and 1, making it useful for binary classification. Given by:

  \[
  f(x) = \frac{1}{1 + e^{-x}}
  \]

- **ReLU (Rectified Linear Unit)**: Outputs the input if it is positive, otherwise it outputs zero. It is widely used in hidden layers for its efficiency:

  \[
  f(x) = \max(0, x)
  \]

- **Tanh Function**: Maps inputs to a value between -1 and 1, providing centered output. Given by:

  \[
  f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]

- **Softmax Function**: Used for multi-class classification, it outputs a probability distribution over multiple classes.

### **Feedforward Neural Networks**

A **feedforward neural network (FNN)** is the most basic type of artificial neural network, where connections between neurons do not form cycles.

- **Architecture**: FNNs consist of an **input layer**, one or more **hidden layers**, and an **output layer**.
- **Flow of Information**: Information flows in one direction, from the input layer through the hidden layers to the output layer.
- **Forward Propagation**: In forward propagation, each neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function to produce an output.

### **Backpropagation and Gradient Descent**

#### **Backpropagation**
**Backpropagation** is the algorithm used to train neural networks by adjusting weights to minimize the output error.

- **Error Calculation**: The difference between the predicted output and the actual output is computed using a loss function (e.g., Mean Squared Error).
- **Chain Rule**: Backpropagation applies the **chain rule** to compute the gradient of the loss function concerning each weight in the network, starting from the output layer and moving backward.
- **Partial Derivatives**: The gradient helps determine how much each weight contributed to the total error and how it should be adjusted.

#### **Gradient Descent**
**Gradient descent** is an optimization algorithm used to minimize the error of a model by updating its parameters (weights and biases).

- **Objective**: The goal of gradient descent is to find the set of weights that minimizes the loss function.
- **Learning Rate (\(\alpha\))**: Determines the step size for each update. A small learning rate results in slow convergence, while a large learning rate can cause the model to overshoot the minimum.
- **Gradient Update Rule**: Each weight is updated using the formula:

  \[
  w_{new} = w_{old} - \alpha \cdot \frac{\partial L}{\partial w}
  \]

  where \( L \) is the loss function and \(\frac{\partial L}{\partial w}\) is the gradient of the loss function with respect to the weight.
- **Variants**: There are several variants of gradient descent:
  - **Batch Gradient Descent**: Uses the entire dataset to compute the gradient.
  - **Stochastic Gradient Descent (SGD)**: Updates weights using a single training example at each iteration.
  - **Mini-batch Gradient Descent**: Uses small batches of the training dataset for each weight update.


# All Activation functions 

### **Activation Functions in Neural Networks: Overview and Use Cases**

Activation functions play a crucial role in determining the output of a neural network and allowing it to learn complex relationships. Here, we discuss the common activation functions along with when to use each of them.

---

### **1. Step Function**

- **Definition**: A binary activation function that outputs either 0 or 1 depending on whether the input exceeds a certain threshold.

  \[
  f(x) = 
  \begin{cases} 
      1 & \text{if } x > 0 \\
      0 & \text{otherwise}
  \end{cases}
  \]

- **When to Use**:
  - **Early Neural Networks**: This was used in perceptrons for simple, binary classification tasks.
  - **Limitations**: It is not differentiable, which makes it unsuitable for modern neural networks that rely on gradient-based optimization techniques. It is rarely used today.

---

### **2. Sigmoid Function**

- **Definition**: Maps the input to a value between 0 and 1, making it suitable for modeling probabilities.

  \[
  f(x) = \frac{1}{1 + e^{-x}}
  \]

- **When to Use**:
  - **Binary Classification**: Use it in the output layer when dealing with a binary classification problem since it outputs values between 0 and 1.
  - **Hidden Layers (Rarely)**: Can be used in hidden layers, but the vanishing gradient issue makes it less efficient in deep networks.
  
- **Limitations**:
  - **Vanishing Gradient Problem**: Large or small input values lead to very small gradients, which makes training slower.
  - **Not Zero-centered**: Outputs are always positive, which can affect optimization efficiency.

---

### **3. Tanh Function (Hyperbolic Tangent)**

- **Definition**: Maps the input to a value between -1 and 1, centered around zero.

  \[
  f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]

- **When to Use**:
  - **Hidden Layers in Shallow Networks**: Preferable over sigmoid in hidden layers as it is zero-centered, allowing for better convergence.
  - **Feature Scaling**: When data is centered around zero, tanh often provides better gradient flow and faster convergence.

- **Limitations**:
  - **Vanishing Gradient**: Similar to sigmoid, tanh can saturate for large input values, leading to a vanishing gradient issue.

---

### **4. ReLU (Rectified Linear Unit)**

- **Definition**: Outputs the input directly if it is positive; otherwise, it outputs zero.

  \[
  f(x) = \max(0, x)
  \]

- **When to Use**:
  - **Hidden Layers in Deep Neural Networks**: ReLU is the most popular choice due to its computational efficiency and ability to avoid saturation for positive inputs.
  - **Image Processing**: It is often used in CNNs (Convolutional Neural Networks) because it accelerates the convergence of training.
  
- **Limitations**:
  - **Dying ReLU Problem**: Neurons may get stuck outputting 0 for negative inputs, effectively "dying" and ceasing to learn.
  
---

### **5. Leaky ReLU**

- **Definition**: A variant of ReLU that allows a small, non-zero gradient for negative inputs.

  \[
  f(x) = 
  \begin{cases} 
      x & \text{if } x > 0 \\
      \alpha x & \text{if } x \leq 0
  \end{cases}
  \]

  where \(\alpha\) is a small positive value (e.g., 0.01).

- **When to Use**:
  - **Deep Neural Networks with Risk of Dying ReLU**: Use Leaky ReLU to mitigate the dying ReLU problem and allow small updates for negative inputs.
  - **Stable Training**: When ReLU results in too many dead neurons, switching to Leaky ReLU can be beneficial.

- **Limitations**:
  - **Fixed Slope for Negative Inputs**: The negative slope is constant, which may not be ideal for all tasks.

---

### **6. Parametric ReLU (PReLU)**

- **Definition**: Similar to Leaky ReLU but with the slope of the negative part learned during training.

  \[
  f(x) = 
  \begin{cases} 
      x & \text{if } x > 0 \\
      \alpha x & \text{if } x \leq 0
  \end{cases}
  \]

  where \(\alpha\) is a parameter learned during training.

- **When to Use**:
  - **Deep Learning Architectures**: Use PReLU when you want the network to learn the most suitable negative slope instead of using a fixed value, which can improve accuracy.
  - **Customizable Negative Region**: When the fixed slope of Leaky ReLU is not yielding good results.

---

### **7. Swish**

- **Definition**: A smooth, non-monotonic activation function that has been shown to outperform ReLU in some cases.

  \[
  f(x) = \frac{x}{1 + e^{-x}}
  \]

- **When to Use**:
  - **Deep Learning Models with Complex Patterns**: Swish is useful when the network benefits from non-monotonic properties, often yielding better accuracy compared to ReLU.
  - **Large Neural Networks**: Can improve performance in deep models, such as transformers.

- **Limitations**:
  - **Computational Complexity**: It is more computationally expensive compared to ReLU.

---

### **8. Softmax Function**

- **Definition**: Used to convert a vector of values into a probability distribution, with each value between 0 and 1 and the sum equal to 1.

  \[
  f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
  \]

- **When to Use**:
  - **Output Layer for Multi-Class Classification**: When the problem involves multiple classes (e.g., image classification with more than two classes), softmax is ideal for generating probabilities for each class.
  - **Normalized Output**: Use when you need the model to output a normalized probability distribution across classes.

- **Limitations**:
  - **Exponentiation Can Cause Overflow**: Softmax can be numerically unstable for very large values, and hence normalization techniques are often used.

---

### **9. GELU (Gaussian Error Linear Unit)**

- **Definition**: Uses a Gaussian cumulative distribution function to decide whether to output the input or not.

  \[
  f(x) = x \cdot \Phi(x)
  \]

  where \(\Phi(x)\) is the cumulative distribution function of a Gaussian distribution.

- **When to Use**:
  - **NLP Models**: GELU has been widely used in models like BERT for natural language processing (NLP) due to its probabilistic behavior.
  - **Transformer Networks**: Suitable for complex networks that benefit from a smoother activation compared to ReLU.

---

### **Summary of Activation Functions and Use Cases**

| Activation Function | Range           | Use Case                     | Key Characteristics                  |
|---------------------|-----------------|------------------------------|--------------------------------------|
| Step Function       | (0, 1) or (-1, 1)| Simple binary classification | Rarely used, not differentiable      |
| Sigmoid             | (0, 1)          | Output for binary classification | Vanishing gradient issue            |
| Tanh                | (-1, 1)         | Hidden layers for zero-centered data | Zero-centered, better than sigmoid  |
| ReLU                | [0, ∞)          | Hidden layers in deep networks | Dying ReLU problem                  |
| Leaky ReLU          | (-∞, ∞)         | Deep networks to avoid dying ReLU | Allows gradient for negative inputs |
| PReLU               | (-∞, ∞)         | Adaptive negative slope in deep models | Learnable negative slope           |
| Swish               | (-∞, ∞)         | Deep learning models requiring non-monotonicity | Smooth gradients, better performance|
| Softmax             | (0, 1)          | Output for multi-class classification | Generates probability distribution |
| GELU                | (-∞, ∞)         | NLP and transformer networks | Probabilistic output, smooth       |

# Cost functions

### **Cost Functions in Deep Learning: Overview and Use Cases**

Cost functions, also known as loss functions, measure how well a neural network model's predictions match the actual outcomes. Different tasks require different types of cost functions. Here, we discuss the common cost functions used in deep learning and when to use each one.

---

### **1. Mean Squared Error (MSE)**

- **Definition**: Measures the average squared difference between predicted values and actual values.

  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]

  where \(y_i\) is the actual value and \(\hat{y}_i\) is the predicted value.

- **When to Use**:
  - **Regression Tasks**: Use MSE when predicting continuous values, such as predicting house prices or stock values.
  - **Benefits**: Penalizes large errors more than smaller ones, making it sensitive to outliers.
  
- **Limitations**:
  - **Sensitive to Outliers**: Since errors are squared, MSE may give too much weight to outliers.

---

### **2. Mean Absolute Error (MAE)**

- **Definition**: Measures the average absolute difference between the predicted values and the actual values.

  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]

- **When to Use**:
  - **Robust Regression**: Use MAE when you want to reduce the impact of outliers compared to MSE.
  - **Regression Tasks**: Suitable for predicting continuous variables where errors are equally weighted.

- **Limitations**:
  - **Less Sensitive to Large Errors**: Unlike MSE, it treats all errors equally.

---

### **3. Huber Loss**

- **Definition**: A combination of MSE and MAE that is less sensitive to outliers than MSE but still differentiable at all points.

  \[
  L_{\delta}(y, \hat{y}) = 
  \begin{cases} 
      \frac{1}{2} (y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\
      \delta \cdot (|y - \hat{y}| - \frac{1}{2} \delta) & \text{otherwise}
  \end{cases}
  \]

- **When to Use**:
  - **Regression with Outliers**: Use when dealing with data containing outliers that may affect the training process.
  - **Smooth Transition**: Provides a smooth transition between MSE and MAE.

- **Limitations**:
  - **Parameter Tuning**: Requires tuning the hyperparameter \(\delta\).

---

### **4. Cross-Entropy Loss (Log Loss)**

- **Definition**: Measures the difference between two probability distributions, typically the predicted and actual class probabilities.

  \[
  L = -\frac{1}{n} \sum_{i=1}^{n} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)
  \]

- **When to Use**:
  - **Binary Classification**: Use Binary Cross-Entropy Loss for binary classification tasks.
  - **Multi-Class Classification**: Use Categorical Cross-Entropy Loss when dealing with multiple classes. If each sample belongs to one of multiple classes, use softmax activation with categorical cross-entropy.

- **Limitations**:
  - **High Penalty for Incorrect Predictions**: It penalizes incorrect predictions heavily, especially if the confidence is high.

---

### **5. Kullback-Leibler Divergence (KL Divergence)**

- **Definition**: Measures how one probability distribution diverges from a reference distribution.

  \[
  D_{\text{KL}}(P || Q) = \sum_{i} P(i) \log\left(\frac{P(i)}{Q(i)}\right)
  \]

- **When to Use**:
  - **Probabilistic Models**: Useful when you need to measure the difference between two probability distributions, such as in generative models.
  - **Autoencoders**: Often used in Variational Autoencoders (VAEs) to make the learned distribution approximate the target distribution.

- **Limitations**:
  - **Asymmetry**: KL divergence is not symmetric, meaning \(D_{\text{KL}}(P || Q) \neq D_{\text{KL}}(Q || P)\).

---

### **6. Hinge Loss**

- **Definition**: A loss function used for "maximum-margin" classification, often used in Support Vector Machines (SVMs).

  \[
  L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})
  \]

  where \(y \in \{-1, 1\}\) and \(\hat{y}\) is the predicted output.

- **When to Use**:
  - **Binary Classification (SVMs)**: Used for binary classification tasks, especially in SVMs to ensure a margin between classes.
  - **Deep Learning for Classification**: Can be used when you want to maximize the margin of classification.

- **Limitations**:
  - **Not Suitable for Probabilities**: Not used for probabilistic output since it is designed for margin-based classification.

---

### **7. Squared Hinge Loss**

- **Definition**: A variant of hinge loss that squares the hinge loss value.

  \[
  L(y, \hat{y}) = (\max(0, 1 - y \cdot \hat{y}))^2
  \]

- **When to Use**:
  - **Binary Classification with Stronger Penalty**: When you want to penalize larger errors more strongly than standard hinge loss.

---

### **8. Poisson Loss**

- **Definition**: Used for modeling count data. It measures the difference between predicted and actual counts.

  \[
  L(y, \hat{y}) = \hat{y} - y \log(\hat{y})
  \]

- **When to Use**:
  - **Count Data Regression**: Use when dealing with count-based data, such as predicting the number of occurrences of an event.

---

### **9. Categorical Hinge Loss**

- **Definition**: A loss function used for multi-class classification tasks.

  \[
  L(y, \hat{y}) = \max(0, \hat{y}_{\text{negative}} - \hat{y}_{\text{positive}} + \Delta)
  \]

  where \(\Delta\) is the margin, typically set to 1.

- **When to Use**:
  - **Multi-Class Classification with Margin**: Suitable for multi-class classification tasks where you want to maximize the margin between the predicted class and other classes.

---

### **10. Cosine Proximity Loss**

- **Definition**: Measures the cosine similarity between the actual and predicted outputs.

  \[
  L(y, \hat{y}) = -\frac{y \cdot \hat{y}}{\|y\| \|\hat{y}\|}
  \]

- **When to Use**:
  - **Similarity-Based Learning**: Use when you want to maximize the similarity between the predicted and actual values, such as in text or image similarity tasks.

- **Limitations**:
  - **Does Not Directly Minimize Error**: Instead, it focuses on direction rather than magnitude.

---

### **Summary of Cost Functions and Use Cases**

| Cost Function            | Use Case                                  | Characteristics                                |
|--------------------------|-------------------------------------------|------------------------------------------------|
| Mean Squared Error (MSE) | Regression                                | Sensitive to outliers, penalizes larger errors |
| Mean Absolute Error (MAE)| Regression                                | Less sensitive to outliers                     |
| Huber Loss               | Robust Regression                         | Balances MSE and MAE                           |
| Cross-Entropy Loss       | Classification (Binary & Multi-Class)     | High penalty for incorrect predictions         |
| KL Divergence            | Probabilistic Models, VAEs                | Measures distribution divergence               |
| Hinge Loss               | Binary Classification (SVM)               | Maximizes margin between classes               |
| Squared Hinge Loss       | Binary Classification with Strong Penalty| Stronger penalty on large errors               |
| Poisson Loss             | Count Data Regression                     | Models count data                              |
| Categorical Hinge Loss   | Multi-Class Classification                | Maximizes margin for multi-class               |
| Cosine Proximity Loss    | Similarity-Based Learning                 | Maximizes directional similarity               |

The selection of a cost function depends on the nature of the problem being solved. For regression tasks, MSE and MAE are most commonly used, while cross-entropy is the preferred choice for classification problems. For data with outliers, Huber Loss can provide a good balance, and for specific requirements like count data or similarity-based tasks, Poisson Loss or Cosine Proximity Loss can be employed.

---
# Optimisers
---

Optimizers are algorithms or methods used to update the parameters (weights) of a neural network in order to minimize the cost function and improve model performance. Different optimizers have their own advantages, limitations, and suitable use cases. Let's discuss various optimizers, their mathematical formulations, working mechanisms, and limitations.

### 1. **Gradient Descent**
Gradient Descent is the most basic optimizer, which adjusts the weights by moving in the direction of the negative gradient of the cost function.

#### **Types:**
1. **Batch Gradient Descent**: Uses the entire training dataset for each step.
   - **Update Rule**: 
     \[
     \theta := \theta - \eta \cdot \frac{1}{n} \sum_{i=1}^n \nabla_\theta L(f(x_i), y_i)
     \]
   - **Pros**: Converges to the global minimum in convex problems.
   - **Cons**: Slow, not suitable for large datasets.

2. **Stochastic Gradient Descent (SGD)**: Uses one sample at each iteration.
   - **Update Rule**: 
     \[
     \theta := \theta - \eta \cdot \nabla_\theta L(f(x_i), y_i)
     \]
   - **Pros**: Faster, suitable for large datasets.
   - **Cons**: May oscillate, not reach the minimum precisely.

3. **Mini-Batch Gradient Descent**: Uses a subset (batch) of the training dataset.
   - **Update Rule**: 
     \[
     \theta := \theta - \eta \cdot \frac{1}{m} \sum_{i=1}^m \nabla_\theta L(f(x_i), y_i)
     \]
   - **Pros**: A balance between Batch and SGD, reduces variance.
   - **Cons**: May still get stuck in local minima.

### 2. **Momentum**
Momentum is an extension of Gradient Descent that helps to accelerate convergence by adding a velocity term.

- **Update Rule**:
  \[
  v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta L
  \]
  \[
  \theta := \theta - \eta v_t
  \]
- **Working**: Uses an exponential moving average of past gradients to dampen oscillations.
- **When to Use**: Suitable for loss surfaces with lots of shallow local minima.
- **Limitations**: Requires tuning of the momentum parameter (\(\beta\)), can overshoot the minimum if not set properly.

### 3. **Nesterov Accelerated Gradient (NAG)**
Nesterov Momentum looks ahead at the future position of the parameter before applying momentum.

- **Update Rule**:
  \[
  v_t = \beta v_{t-1} + \eta \nabla_\theta L(\theta - \beta v_{t-1})
  \]
  \[
  \theta := \theta - v_t
  \]
- **Working**: Estimates gradients at a future position to adjust faster.
- **When to Use**: Helps with faster convergence when compared to vanilla momentum.
- **Limitations**: More computationally intensive, requires careful tuning.

### 4. **Adagrad (Adaptive Gradient Algorithm)**
Adagrad adjusts the learning rate for each parameter based on past gradients.

- **Update Rule**:
  \[
  \theta := \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_\theta L
  \]
  where \( G_t \) is the sum of squares of past gradients.
- **Working**: Parameters that have smaller gradients get larger learning rates, and vice versa.
- **When to Use**: Suitable for sparse data problems like NLP.
- **Limitations**: Accumulated squared gradients keep growing, leading to diminishing learning rates.

### 5. **RMSprop (Root Mean Square Propagation)**
RMSprop modifies Adagrad by limiting the accumulation of past squared gradients.

- **Update Rule**:
  \[
  E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2
  \]
  \[
  \theta := \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla_\theta L
  \]
- **Working**: Applies a decaying average to the squared gradients.
- **When to Use**: Suitable for non-stationary problems like RNNs.
- **Limitations**: Requires tuning of the decay parameter.

### 6. **Adam (Adaptive Moment Estimation)**
Adam combines the benefits of RMSprop and momentum by using estimates of both first and second moments of the gradients.

- **Update Rule**:
  - First Moment Estimate:
    \[
    m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta L
    \]
  - Second Moment Estimate:
    \[
    v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta L)^2
    \]
  - Bias-Corrected Estimates:
    \[
    \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
    \]
  - Parameter Update:
    \[
    \theta := \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
    \]
- **Working**: Adam adjusts the learning rate based on first and second moments of gradients.
- **When to Use**: Suitable for most deep learning tasks due to faster convergence.
- **Limitations**: Sensitive to hyperparameter values, particularly \(\beta_1\), \(\beta_2\), and \(\eta\).

### 7. **AdaDelta**
AdaDelta is an extension of Adagrad that seeks to reduce the diminishing learning rate issue.

- **Update Rule**:
  \[
  E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2
  \]
  \[
  \Delta \theta_t = - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \nabla_\theta L
  \]
- **Working**: Unlike Adagrad, it limits the window of accumulated past gradients.
- **When to Use**: Suitable for problems where diminishing learning rates are problematic.
- **Limitations**: More complex to implement and requires more memory.

### 8. **Adamax**
Adamax is a variant of Adam based on the infinity norm, making it more robust in certain situations.

- **Update Rule**:
  - Similar to Adam but the denominator is replaced by the maximum of past gradients.
- **Working**: Uses the infinity norm for better stability.
- **When to Use**: Can be more stable than Adam, especially when dealing with high-dimensional data.
- **Limitations**: Similar to Adam but computationally cheaper.

### 9. **Nadam (Nesterov-accelerated Adam)**
Nadam combines Adam with Nesterov momentum.

- **Update Rule**:
  - Applies Nesterov momentum to the Adam optimizer.
- **Working**: Adjusts the gradient by considering a future position before applying it.
- **When to Use**: Can improve convergence speed in some scenarios.
- **Limitations**: Computationally intensive and hyperparameter-sensitive.

### **Choosing the Right Optimizer**

1. **SGD**: Use when you need simplicity and have a large dataset. It is useful when precise convergence is needed and computational resources are limited.
  
2. **Momentum/NAG**: Use when facing issues with slow convergence and oscillations, particularly in deep neural networks.

3. **Adagrad**: Use when working with sparse data where some parameters require more frequent updates.

4. **RMSprop**: Ideal for non-stationary problems and recurrent neural networks (RNNs).

5. **Adam**: Default choice for most problems. Works well for large datasets and in situations where we expect noisy gradients.

6. **AdaDelta**: Useful when diminishing learning rates from Adagrad are problematic. Works well in deep networks.

7. **Adamax**: Use when dealing with high-dimensional data where Adam may become unstable.

8. **Nadam**: Useful when a combination of momentum and adaptive learning rates is required for faster convergence.

### **Summary**
Each optimizer has its pros and cons, and their effectiveness largely depends on the problem type, data structure, and computational resources available. Adam is often a good default choice due to its adaptive nature and efficient convergence, while SGD with momentum or NAG can be effective for smaller-scale tasks where control over convergence is crucial. The correct choice requires experimentation and tuning based on the specific characteristics of the problem at hand.
