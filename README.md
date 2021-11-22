# Time-Series-Analysis-of-Continuous-Flow-Process

## Name : Aditi Ganesh Joshi

**Roll No. :** 180020010

## System

This project aims to predict outputs of a **continuous-flow manufacturing process** in an actual production line near Detroit, Michigan. Dataset has been directly obtained from the website[1]. In the first stage, the process has **3** machines operating in parallel. The outputs from those machines are combined and measured at **15** different locations. This is the primary prediction to be made and the measurements are noisy. This is a **multivariate regression problem**. It is a **univariate time series with exogenous variables.**

## Input-Output

Naming conventions:

1. .C.Setpoint - Setpoint for Controlled variable
2. ~C.Actual - Actual value of Controlled variable
3. .U.Actual - Actual value of Uncontrolled variable
4. Others - Environmental or raw material variables, States/events, etc.

The dataset contains the following columns:

0 - Timestamp

1 to 2 - Factory ambient conditions

3 to 6 - First stage, Machine 1, raw material properties (material going into Machine 1)

7 to 14 - First stage, Machine 1 process variables

15 to 18 - First stage, Machine 2, raw material properties (material going into Machine 2)

19 to 26 - First stage, Machine 2 process variables

27 to 30 - First stage, Machine 3, raw material properties (material going into Machine 3)

31 to 38 - First stage, Machine 3 process variables

39 to 41 - Combiner stage process parameters. Here we combine the outputs from Machines 1, 2, and 3.

42 - Measurement 0 of Stage 1 output

43 - Setpoint of Measurement 0 of Stage 1 output

Hence, there are **41** input/independent variables and **1** output/dependent variable. Moreover, the output(42 - Measurement 0 of Stage 1 output) is a time series.The dataset size is **14088 x 43**

**Data Preparation and Visualization**

The following steps are followed while preprocessing the data:

1. Null values are checked, and such rows are removed.
2. Next, the **output versus timestamp** plot is plotted to observe the nature of the time series.



The time series is **mean-centered** but not white noise(mean=0).

1. **Box plot** of the output is plotted:

Many outliers are observed. These are removed by **interquartile range** evaluation. Points lying outside 1.3\*(interquartile range) are replaced with the mean of the time series. Here, _interquartile range = quartile3 - quartile1_. After replacing the outliers, the box plot obtained has no outliers:



The time series plot is as follows:



1. **Output versus set point** is plotted to determine the presence of a feedback loop in the system.

![](RackMultipart20211122-4-cwp686_html_66ec911c3c7c82a7.png)

The setpoint is constant for most of the points. There are a few exceptions where both output and setpoint are zero. These can be treated as exceptions when the sensor fails to receive the signal for the output. Also, the number of such points(56) is very less than the total number of data points(14088) and can be ignored. Hence, there is **no feedback loop** in the system and we don&#39;t consider setpoint as an input variable.

1. For testing the stationarity of the time series, **Augmented Dickey Fuller(ADF) Test** is performed. The results are as follows:

![](RackMultipart20211122-4-cwp686_html_83671059d8302ef1.png)

The ADF statistic(-3.6335) is lower than the critical value at 1% significance level(-3.4308). Also, the p-value is less than 0.05. This means that we can reject the null hypothesis. Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is **stationary** or does not have time-dependent structure.

1. **Box-cox transformation** is performed. An optimal lambda of 25.45 is obtained. The histogram of the output before and after are as follows:

Before:

![](RackMultipart20211122-4-cwp686_html_a59ea7e41bfac4d7.png)

After:

![](RackMultipart20211122-4-cwp686_html_a6965876a099cb35.png)

The Q-Q plot is as follows:

![](RackMultipart20211122-4-cwp686_html_4e56efae3af73b7d.png)

Since it is very far away from the y=x line, we can conclude that box-cox transformation doesn&#39;t work well with this time series.

1. The time-series is **decomposed** as follows:

y(t) = Level + Trend + Seasonality + Residual

![](RackMultipart20211122-4-cwp686_html_15321f1954d4ed52.png)

No significant trend or seasonality is observed.

1. **Autocorrelation and partial autocorrelation** plots are obtained as follows:

![](RackMultipart20211122-4-cwp686_html_624eba65ea1c3d75.png) ![](RackMultipart20211122-4-cwp686_html_7423d4cde4dfa910.png)

The autocorrelation plot trails off and the partial autocorrelation plot shows a hard cut-off at lag = 2. This observation suggests that an ARx model with lags = p = 1 will be suitable for this time series.

## Solution

1. As suggested by the acf and pacf curves, an **AR**** x **** model with p=1** can be a fit for this problem. The model equation is as follows:

y(t) + a1y(t-1) = e(t) + u(t)

Mean squared error for p = 1 is **3.0653**

Normalized Root Mean Squared error for p = 1 is **4.1988**

The Q-Q plot for residuals is as follows:

![](RackMultipart20211122-4-cwp686_html_ec0b482bc3296f6e.png)

The residuals are not distributed normally.

The autocorrelation of residuals is as follows:

![](RackMultipart20211122-4-cwp686_html_40d23f777c059b03.png)

The residuals are not white noise. Hence, this model is not a good fit.

1. **Recurrent Neural Networks** - These can handle sequential data unlike feedforward Neural Networks. Hence, they are suitable for time-correlated dynamic systems. These networks take into account the time dependence between different samples and train the network through Backpropagation Through Time(BPTT) algorithm.

RNNs don&#39;t have long term memory though. Hence, **Long short-term memory networks(LSTMs)** have been recently developed. An LSTM looks as follows:

![](RackMultipart20211122-4-cwp686_html_65f9709d94b6a4f.png)

1. The cell state Ct helps the network to remember information over a long range.
2. The forget gate allows the network to throw unnecessary information.
3. The input gate decides what new information the cell will store.
4. The tanh layer creates a new candidate for the cell state(Ct~).
5. The cell state is updated as:

![](RackMultipart20211122-4-cwp686_html_b7285ab736333f57.png)

1. Finally, the output is evaluated as:

![](RackMultipart20211122-4-cwp686_html_45ed0b7950c20fee.png)

Tanh is the popular choice for activation function in case of LSTMs. A **single layered LSTM with 20 units** has been employed here:

![](RackMultipart20211122-4-cwp686_html_f43f2accf4cc0baa.png)

The details of the network are as follows:

1. The LSTM layer uses a **tan(h)** activation function.
2. The Dense layer utilizes **ReLU** as its activation function, since this is a regression problem.
3. The optimizer used to solve for minimum loss(mean squared error) is **Adam** optimizer.
4. A **Dropout** layer with a probability of 0.2 is employed to avoid overfitting.

The loss (mean squared error) function is as follows:

![](RackMultipart20211122-4-cwp686_html_72a775f06a934255.png)

Root Mean Squared error is **0.7897**

Normalized Root Mean Squared error is **1.8938**

If we increase the number of units(say, 26), the loss function is as follows:

![](RackMultipart20211122-4-cwp686_html_43bcf50ba953caf3.png)

This plot implies that such a model overfits the data. Hence, the values of errors for the test dataset are high:

Root Mean Squared error is **1.0132**

Normalized Root Mean Squared error is **2.4299**

On increasing the number of units, we increase the complexity of the model and it becomes prone to overfitting. Hence, a model with **20** units in an LSTM is a good fit.

**Conclusion**

1. No feedback loop is present in the given system and hence, this is a **regularization problem.**
2. An ADF Statistic lower than a critical value, with 1% probability (in this case), indicates a **stationary** time series.
3. Box cox transforms are not always effective while converting a time series to a normally distributed series.
4. The seasonal decomposition of the data indicates no significant trend or seasonality.
5. Regularization and reduction in complexity of models help in preventing the overfitting of models.
6. **LSTMs** provide better models for time series data over traditional statistical models like **AR**. In this case, it reduced the normalized root mean squared error by almost **4** times.

**Revisions to be made**

1. Revise the sampling time for better autocorrelations
2. Inspect integrating behaviour- examine coefficient(a1) of ARx model
3. Explore methods to determine order of ARx model
4. Feature engineering- Choose certain features to build more robust model

**References:**

1. Source of data- [https://www.kaggle.com/supergus/multistage-continuousflow-manufacturing-process](https://www.kaggle.com/supergus/multistage-continuousflow-manufacturing-process)
2. [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
3. [https://www.statsmodels.org/stable/index.html](https://www.statsmodels.org/stable/index.html)
4. [https://scipy.org/](https://scipy.org/)
5. [https://matplotlib.org/](https://matplotlib.org/)
6. [https://keras.io/](https://keras.io/)
7. [https://machinelearningmastery.com](https://machinelearningmastery.com/time-series-data-stationary-python/)
8. [https://towardsdatascience.com](https://towardsdatascience.com/lets-forecast-your-time-series-using-classical-approaches-f84eb982212c)
