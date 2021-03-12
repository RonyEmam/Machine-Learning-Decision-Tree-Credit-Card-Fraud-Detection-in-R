---
title: "Untitled"
output: html_document
---

## Credit Card Fraud Detection

The goal of this module is to detect credit card fraudulent transactions, so it's better to be familiar with machine learning algorithms concepts before I start:

1. Logistic regression
2. Decision trees
3. Neural network

Also I have uploaded the dataset that I have used in this module.

Final steps these are the basic libraries that I need them. although there are more libraries needed here. I will mention them all during the journey!

```{r}
# Loading the required libraries, although there are more libraries needed here. I will mention them all during the journey!

library(ggplot2)
library(lattice)
library(caret)
library(ranger)
library(data.table)
```

### 1. First step is importing the dataset, and understand the dataset. What's represents, identify the variables.

Understanding which numbers are continuous also comes in handy when thinking about the type of plot to use to represent your data visually.

It might find some descriptive data analysis as well. It is crucial **to know all the values, observations and missing values.** As it will help later with the outliers and plots.
Even counting the NA values can help with the dataset.

```{r}
df1.0 <- read.csv("creditcard.csv")
head(df1.0, 5)
```


### 2. Manipulate my dataset.

It's an advanced level of understanding the dataset. Any analyst finds that spending the biggest the overall time on doing these two steps. 

As real datasets are really messy and need a lot of attention and efforts. The rest of steps such as picking up the model or visualizing your data is a light typing (visualizing/plotting doesn't take more than 10 lines of coding!).

I have executed the following functions:

is.na() ~ To check the NA values.
is.empty() ~ To check the empty cells from the rapport library. (You need to download the library first)

As you can see I have time variable, 28V variable and amount variable. Also I have class variable. What I mean by variable is a column. Observation is a row. Finally value is the number or output in one cell.

Now I will use summary() function to generate some numerical values. **It is really great to see that I have the basic outputs such as the mean, median, 25% quantile, 75% quantile, min & max values in the dataset.** I can use these results to compare between values or even as a benchmark.

```{r}
summary(df1.0)
```

```{r}
# It's important to visualize the first observations of the dataset. Check the type of each variable whether it's object, numerical value, etc.
df1.0$Amount=scale(df1.0$Amount)
df=df1.0[,-c(1)]
head(df)
```


### 3. Modelling the data I have:

The set. seed() function sets the starting number used to generate a sequence of random numbers – it ensures that you get the same result if you start with that same seed each time you run the same process. For example, if I use the sample() function immediately after setting a seed, I will always get the same sample.

```{r}
# Import the "caTools" library 
library(caTools)
set.seed(123)
```

```{r}
data_smaple = sample.split(df$Class, SplitRatio = 0.80)

train_df = subset(df, data_smaple == TRUE)
test_df = subset(df, data_smaple == FALSE)

```

```{r}
# Find out the dimensions by using the dim() function.
dim(train_df)
dim(test_df)
```
In this section of credit card fraud detection project, I will fit my first model. I will use logistic regression. 

**A logistic regression is** used for modeling the outcome probability of a class such as fraud/not fraud. Then proceed to implement this model on the test data as follows

```{r}
# It's important to use summary() function to see the deviance residuals, intercept values, etc.
# All these values will help us in understanding the linear models later.
logistic_model=glm(Class~.,test_df,family=binomial())
summary(logistic_model)

```

```{r}
# Visualize the logistic model:
plot(logistic_model)
```

In this step, I want to see the performance of the model I have created, so first recall the required libraries then analyize the performance.

```{r}
library(rpart)
library(rpart.plot)
```


```{r}
decision_tree_model <- rpart(df$Class ~ . , df, method = "class")
predicted_val <- predict(decision_tree_model, df, type = "class")
probability <- predict(decision_tree_model, df, type = "prob")
```


```{r}
rpart.plot(decision_tree_model)
```

### 4. Artificial Neural Network:

**act.fct.:** a differentiable function that is used for smoothing the result of the cross product of the covariate or neurons and the weights. 
(more information can be found in this link:  https://www.rdocumentation.org/packages/neuralnet/versions/1.44.2/topics/neuralnet)

```{r}
# Import [neuralnet] library in order to show the data based on the historical data 
# and are able to perform classification on the input data.
library(neuralnet)

# Creating my ANN_model using neuralent function. I have marked linear.output as FALSE because 
# act.fct should not be applied to the output neurons
ANN_model = neuralnet(Class ~ ., train_df, linear.output = FALSE)
plot(ANN_model)
```


Now, in the case of Artificial Neural Networks, there is a range of values that is between 1 and 0. Then set a threshold as 0.5, that is, values above 0.5 will correspond to 1 and the rest will be 0. I will implement this as follows:
 
```{r}
predANN = compute(ANN_model, test_df)
resultANN = predANN$net.result
resultANN = ifelse(resultANN > 0.5, 1, 0)
```

### 5. Gradient Boosting (GBM):

is a popular machine learning algorithm that is used to perform classification and regression tasks. This model comprises of several underlying ensemble models like weak decision trees. 
These decision trees combine together to form a strong model of gradient boosting. I will implement gradient descent algorithm in the model

```{r}
library(gbm, quietly = TRUE)
```


```{r}
# Use the GBM model

system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_df, test_df)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.0
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_df) / (nrow(train_df) + nrow(test_df))
                   )
)
```


```{r}
# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
```

```{r}
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)

# Visualize the GBM model
plot(model_gbm)
```

```{r}
library(pROC)
```


```{r}
# Plot and calculate AUC on test data
gbm_test = predict(model_gbm, newdata = test_df, n.trees = gbm.iter)
gbm_auc = roc(test_df$Class, gbm_test, plot = TRUE, col = "red")
```


```{r}
print(gbm_auc)
```

