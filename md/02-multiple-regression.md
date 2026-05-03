# Lecture 2: Multiple Regression

## Introduction

The linear model seen in the previous lecture can be extended to multiple variables. This is useful if we want to predict the price of a house based on more than just one variable, e.g. the size **and** the number of rooms.

Otherwise the model stays the same. This time however we want to fit a hyperplane to the data.

### Multiple Regression Model

$$
y = \beta_0 + \sum_{i=1}^k \beta_i x_i
$$

where $\beta_i$ are the coefficients and $x_i$ are the variables.

### Assumptions of the model

Similarly the model above only works if the data behaves as: $y_i=\beta_0 + \sum_{i=1}^k \beta_i x_i + \epsilon_i$ and $\epsilon_i \sim N(0, \sigma^2)$.

## Least squares introduction

The least squares method can be extended to multiple variables. The only difference is that we now have to find $\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_k$ to minimize the sum of squared residuals:

$$
SS_{residual} = \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

For the calculations to work out nicely we need to define higher dimensional random vectors and matrices.

### Random vectors and matrices

A random vector is a vector whose components are random variables. Similarly a random matrix is a matrix whose components are random variables. For example:

$$
V = \begin{pmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{pmatrix} \in \mathbb{R}^n \quad \text{and} \quad M = \begin{pmatrix} X_{11} & X_{12} & \dots & X_{1n} \\ X_{21} & X_{22} & \dots & X_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ X_{m1} & X_{m2} & \dots & X_{mn} \end{pmatrix} \in \mathbb{R}^{m \times n}
$$

#### Expected value

The expected value of such objects is defined as the matrix obtained by taking the expected value of each component. For example:

$$
E(M) = \begin{pmatrix} E(X_{11}) & E(X_{12}) & \dots & E(X_{1n}) \\ E(X_{21}) & E(X_{22}) & \dots & E(X_{2n}) \\ \vdots & \vdots & \ddots & \vdots \\ E(X_{m1}) & E(X_{m2}) & \dots & E(X_{mn}) \end{pmatrix} \in \mathbb{R}^{m \times n}
$$

#### Variance matrix

Calculating the variance is a bit more complicated. Lets start with the variance of a random vector:

$$
V = \begin{pmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{pmatrix} \quad \text{with} \quad \mu= E(V) = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_n \end{pmatrix}
$$

where $\mu$ is the mean of the random vector. The variance matrix is defined as:

$$
\begin{aligned}
\Sigma = Var(V) &= E((V - \mu)(V - \mu)^T) \\
      &= \begin{pmatrix}
        E[(X_1 - \mu_1)^2] & E[(X_1 - \mu_1)(X_2 - \mu_2)] & \dots & E[(X_1 - \mu_1)(X_n - \mu_n)] \\
        E[(X_2 - \mu_2)(X_1 - \mu_1)] & E[(X_2 - \mu_2)^2] & \dots & E[(X_2 - \mu_2)(X_n - \mu_n)] \\
        \vdots & \vdots & \ddots & \vdots \\
        E[(X_n - \mu_n)(X_1 - \mu_1)] & E[(X_n - \mu_n)(X_2 - \mu_2)] & \dots & E[(X_n - \mu_n)^2]
      \end{pmatrix} \\
\end{aligned}
$$

The matrix $\Sigma$ is symmetric and positive semi-definite. The diagonal elements are the variances of the components of $V$ and the off-diagonal elements are the covariances of the components of $V$. If the matrix is diagonal, then the components of $V$ are independent.

#### Properties of random objects

Let $A$ be a constant matrix and $y$ a random vector. Then $Ay$ is also a random vector and:

+ $E(Ay) = AE(y)$
+ $Var(Ay) = AVar(y)A^T$
+ if $y$ is normally distributed, then $Ay$ is also normally distributed

### Least squares model in matrix form

To represent the assumptions of the model in matrix form we can write:

$$
\begin{aligned}
Y &= X\beta + \epsilon \\
\end{aligned}
$$

The matrix $X$ is called the design matrix and is defined as:

$$
X = \begin{pmatrix} 1 & x_{11} & x_{12} & \dots & x_{1k} \\ 1 & x_{21} & x_{22} & \dots & x_{2k} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \dots & x_{nk} \end{pmatrix}
\in \mathbb{R}^{n \times (k+1)} \quad \text{and} \quad \beta = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_k \end{pmatrix} \in \mathbb{R}^{(k)}
$$

Adding the constant column of $1$'s to the design matrix allows to simplify the notation. The vector $\epsilon$ is defined as:

$$
\epsilon = \begin{pmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{pmatrix} \quad \text{with} \quad \epsilon_i \sim N(0, \sigma^2)
$$

To simplify the notation we can write: $\epsilon \sim N(0, \sigma^2 I_n)$ where $I_n$ is the identity matrix of size $n$. The expected value of $\epsilon$ is $E(\epsilon) = 0$ and the variance matrix is $Var(\epsilon) = \sigma^2 I_n$.

### Least squares solution

If we put the least squares model in matrix form we get:

$$
\begin{aligned}
SS_{residual} &= \sum_{i=1}^n (y_i - \hat{y}_i)^2 \\
              &= \sum_{i=1}^n (y_i - \hat{\beta}_0 - \sum_{j=1}^k \hat{\beta}_j x_{ij})^2 \\
              &= (Y - X\hat{\beta})^T (Y - X\hat{\beta}) \\
              &= Y^T Y - Y^T X \hat{\beta} - \hat{\beta}^T X^T Y + \hat{\beta}^T X^T X \hat{\beta} \\
              &= Y^T Y - 2 \hat{\beta}^T X^T Y + \hat{\beta}^T X^T X \hat{\beta} \\
\end{aligned}
$$

To find the minimum we can take the derivative with respect to $\hat{\beta}$ and set it to $0$:

$$
\begin{aligned}
\frac{\partial SS_{residual}}{\partial \hat{\beta}} &= -2 X^T Y + 2 X^T X \hat{\beta} \\
X^T Y &= X^T X \hat{\beta} \\
\hat{\beta} &= (X^T X)^{-1} X^T Y \\
\end{aligned}
$$

This only works if $X^T X$ is invertible. This is the case if the columns of $X$ are linearly independent.

### Hat matrix

Using the multiple regression model we can predict the value of $\hat{y}$ for a new observation $x^*$:

$$
\hat{y} = X \hat{\beta} = \underbrace{X (X^T X)^{-1} X^T}_{H} y = H y
$$

where $H$ is called the hat matrix. $H$ it is a so called projection matrix, i.e. $H$ projects $y$ onto the linear subspace spanned by the columns of $X$.

$\hat{y}$ is the vector of predicted values for the data points in $X$.

All projection matrices have the following properties:

+ $H$ is symmetric
+ $H$ is idempotent, i.e. $H^2 = H$

It is called the hat matrix because it puts a hat on $y$ to indicate that it is a prediction.

### Residuals

The residuals are defined as:

$$
e = y - \hat{y} = y - H y = (I_n - H) y
$$

$I_n - H$ is also a projection matrix. It projects $y$ onto the linear subspace orthogonal to the linear subspace spanned by the columns of $X$.

Rearranging the equation above we get:

$$
y = \hat{y} + e
$$

This means that the observed value $y$ can be written as the sum of the predicted value $\hat{y}$ and the orthogonal residual $e$.

### Maximum likelihood

Using the same principle as in the previous lecture we can show that the least squares solution is the maximum likelihood solution. For this we first need to define the data model in a probabilistic way:

$$
\begin{aligned}
y_i &= X \beta + \epsilon \quad \text{with} \quad \epsilon \sim N(0, \sigma^2 I_n) \\
\implies y_i &\sim N(\mu_i, \sigma^2) \quad \text{with} \quad \mu_i = X_i \beta = \beta_0 + \sum_{j=1}^k \beta_j x_{ij} \\
\implies y &\sim N(X\beta, \sigma^2 I_n) \quad
\end{aligned}
$$

This means that the probability density function of $y_i$ is:

$$
f(y_i) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(y_i - \mu_i)^2}{2 \sigma^2} \right)
$$

The likelihood function is defined as:

$$
\begin{aligned}
L(\beta, \sigma^2| y_1, \dots, y_n) &= \prod_{i=1}^n f(y_i) \\
                   &= \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(y_i - \mu_i)^2}{2 \sigma^2} \right) \\
                   &= \frac{1}{\left(\sqrt{2 \pi \sigma^2}\right)^{n}} \exp \left( - \frac{1}{2 \sigma^2} \sum_{i=1}^n (y_i - \mu_i)^2 \right) \\
                   &= \frac{1}{\left(\sqrt{2 \pi \sigma^2}\right)^{n}} \exp \left( - \frac{1}{2 \sigma^2} (y - X \beta)^T (y - X \beta) \right) \\
\end{aligned}
$$

### Unbiased estimator of $\sigma^2$ in multiple regression

Maximizing the likelihood function with respect to $\sigma^2$ yields:

$$
s^2 = \frac{1}{n-k-1} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

where $k$ is the number of predictors without the intercept. This is an unbiased estimator of $\sigma^2$.

## Unbiased estimator of $\beta$ in multiple regression

Maximizing the likelihood function with respect to $\beta$ yields:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

This is an unbiased estimator of $\beta$.

## Variance of $\hat{\beta}$ in multiple regression

The variance of $\hat{\beta}$ is defined as:

$$
Var(\hat{\beta}) = \sigma^2 (X^T X)^{-1}
$$

## Anova table for multiple regression

The ANOVA table in multiple regression is similar to the one in simple regression.

![ANOVA table](images/ANOVA-table.png)

However there exist some convenient formulas to calculate the values in the table in matrix form:

$$
\begin{aligned}
SS_{total} &= y^T y - n \bar{y}^2 \\
SS_{residual} &= y^T y - \hat{\beta}^T X^T X \hat{\beta} \\
SS_{regression} &= \hat{\beta}^T X^T X \hat{\beta} - n \bar{y}^2 \\
\end{aligned}
$$

where $SS_{total} = SS_{regression} + SS_{residual}$ still holds.

## F-test

### F-test hypothesis

The F-statistic is used to test the significance of the model. In particular it tests the null hypothesis $H_0: \beta_1 = \beta_2 = \dots = \beta_k = 0$. Therefore it checks if there is any linear relationship between the response variable $y$ and any of the predictors $x_1, x_2, \dots, x_k$.

### F-test statistic

The F-statistic is defined as:

$$
F = \frac{MS_{regression}}{MS_{residual}} = \frac{SS_{regression} / k}{SS_{residual} / (n-k-1)} = \frac{SS_{regression}}{s^2} \sim F_{k, n-k-1}
$$

where $MS$ stands for mean square and $SS$ for sum of squares. The F-statistic follows an F-distribution with $k$ and $n-k-1$ degrees of freedom.

The Null hypothesis $H_0$ is rejected if

+ $F > F_{k, n-k-1, 1-\alpha}$
+ $p = P(Z > F) < \alpha$ where $Z \sim F_{k, n-k-1}$

where $p$ is the p-value of the F-statistic.

## T-test

### T-test hypothesis

The T-statistic is used to test the significance of each predictor. In particular it tests the null hypothesis $H_0: \beta_i = 0$ for each predictor $x_i$.

### T-test statistic

The T-statistic in multiple regression is defined as:

$$
t_i = \frac{\hat{\beta}_i}{se(\hat{\beta}_i)} = \frac{\hat{\beta}_i}{\sqrt{Var(\hat{\beta}_i)}} \sim t_{n-k-1}
$$

where $se$ stands for standard error. The T-statistic follows a T-distribution with $n-k-1$ degrees of freedom.

The Null hypothesis $H_0$ is rejected if

+ $|t_i| > t_{n-k-1, 1-\alpha/2}$
+ $p = 2 P(Z > |t_i|) < \alpha$ where $Z \sim t_{n-k-1}$

where $p$ is the p-value of the T-statistic.

### Confidence intervals

The confidence interval for $\beta_i$ is defined as:

$$
\hat{\beta}_i \pm t_{n-k-1, 1-\alpha/2} se(\hat{\beta}_i)
$$

## Association vs. causation

The multiple regression model can be used to find associations between variables. However it cannot be used to find causations. For example:

+ $x_1$ = number of fire fighters
+ $x_2$ = number of fires
+ $y$ = damage caused by fires

The model $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$ would find a positive association between $x_1$ and $y$ and a negative association between $x_2$ and $y$. However it would be wrong to conclude that hiring more fire fighters would cause more fires and that more fires would cause more damage. In this case the number of fires is the confounding variable causing the association between $x_1$ and $y$.

In general it cannot be differentiated between the following two cases:

+ $X$ causes $Y$
+ $Z$ causes $X$ and $Y$
+ $X$ and $Z$ cause $Y$

## Collinearity

If two or more predictors are highly correlated, then the matrix $X^T X$ is not invertible. This means that the least squares solution cannot be calculated. This is called collinearity.

## Linear hypothesis

If we have a regression model $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_5 x_5 + \beta_6 x_6 +\epsilon$ and we want to test the hypothesis $H_0: \beta_4 = \beta_5 = \beta_6 = 0$ (i.e. all predictors are zero). We can rewrite the model as:

$$
\begin{aligned}
H_0: A \beta &= 0 \quad \text{with} \quad A = \begin{pmatrix} 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{pmatrix}\\
&= \begin{pmatrix} 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \beta_3 \\ \beta_4 \\ \beta_5 \\ \beta_6 \end{pmatrix} =0\\
&= \begin{pmatrix} \beta_4 \\ \beta_5 \\ \beta_6 \end{pmatrix} = 0
\end{aligned}
$$

### F-Test for linear hypothesis

The hypothesis test can be written as:

$$
\begin{aligned}
H_{restricted}: \mu_A &= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3\\
H_{full}: \mu_B &= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_5 x_5 + \beta_6 x_6\\
\end{aligned}
$$

This means that in general the full model has more free parameters than the restricted model and is therefore more flexible. And a better predictor. In particular this means that:

$$
SS_{residual}^ {restricted} \geq SS_{residual}^{full}
$$

The F-statistic is defined as:

$$
F = \frac{(SS_{residual}^{restricted} - SS_{residual}^{full}) / a}{SS_{residual}^{full} / (n-k-1)} \sim F_{a, n-k-1}
$$

where $a$ is the number of restrictions (i.e. the number of rows in $A$ which is $3$ in this case).

The Null hypothesis $H_0$ is rejected either if:

+ $F > F_{a, n-k-1, 1-\alpha}$
+ p = $P(Z > F) < \alpha$ where $Z \sim F_{a, n-k-1}$
