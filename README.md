## Introduction to Data Science and Artificial Intelligence (SC1015)
School of Computer Science and Engineering (SCSE) <br>
Nanyang Technological University (NTU) <br>
Lab: B133 <br>
Team: 3 <br>

Members:
1. Ni Jun Hong Glenn
2. See Qin Yuan
---
### Introduction
The goal of this project:
> To predict the sales of the "Food" category from a time-series dataset using the "sales", "onpromotion", and "holiday" variables. <br>

This repository includes the source code and dataset used in the project, as well as a notebook containing all the code, graphs, and explanations using markdown cells and comments within the code. This README file provides a general overview of the project, including the rationale behind the decisions we made. The original dataset and inspiration can be found [here](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). <br>

---
### Table of Contents
1. [Motivation & Problem Definition](#1-motivation--problem-definition)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Models](#3-model-arima)
4. [Insights & Conclusion](#4-insights--conclusion)
5. [Contributions](#5-contributions)
6. [References](#6-references)

---

### 1. Motivation & Problem Definition
*Motivation*: <br>
Our team is committed to making a positive impact on the world, and we believe that sustainable development is crucial to achieving this goal. That's why we have chosen to focus on Sustainable Development Goal 12, Responsible Consumption and Production. By promoting sustainable consumption and production patterns, reducing waste and pollution, and improving resource efficiency, we can help create a more sustainable future for all.
To achieve this goal, we have identified a specific problem that we aim to tackle - food wastage by grocery stores. Food wastage is a major issue that has significant economic, social, and environmental implications. Where according to the Food and Agriculture Organization of the United Nations (FAO), roughly one-third of all food produced globally is wasted. This leads to unnecessary resource depletion, contributes to greenhouse gas emissions, and causes financial losses for businesses. By reducing food wastage, we can reduce the environmental impact of grocery stores, save resources, and support local communities.

*Problem Definition*: <br>
To address this issue, we will leverage the dataset provided by Corporación Favorita, one of the largest Ecuadorian-based grocery retailers. By analyzing this data, we aim to identify patterns and trends that can help improve inventory management, reduce waste, and promote more sustainable consumption and production patterns. We believe that by using data-driven insights to drive decision-making in grocery stores, we can create a more efficient and sustainable food supply chain.

### 2. Exploratory Data Analysis (EDA)
*Data Preparation*: <br>
To prepare the data for analysis, we extracted "FOOD" related segments from the original dataset and filtered it to only include stores related to Quito. We also converted the data type of the relevant columns to numerical format to enable further analysis. Next, we merged the filtered datasets on the 'date' column to create a single, cleaned dataset that is suitable for analysis. These data preparation steps were taken to ensure that the data is relevant, accurate, and in the correct format for analysis. <br>

*Data Visualisation*: <br>
1. Distribution: Average Sale. 
   - The provided code explores the distribution of average sales across different 'family' categories in the dataset. It creates a grid of 3x3 subplots using matplotlib to display histograms of sales for each 'family' category, as well as a separate histogram for the 'SEAFOOD' category. Seaborn is used to create the histograms, with the 'sales' column as the variable and the kde parameter set to True. The resulting visualizations provide insights into the distribution of average sales across different 'family' categories and can help identify trends and anomalies within the dataset. <br>
   - Observation: The majority of the curves appear narrow, indicating that the data points are closely clustered together for each year between 2014 and 2017. This suggests that the sales within each category are relatively consistent over time, with little variation from year to year.

2. LinePlot
   - The provided code explores the trend of daily and weekly sales across different 'family' categories in the dataset using line plots created with seaborn. The 'sales' column is plotted on the y-axis and the 'date' column is plotted on the x-axis. The line plots provide insights into the trends in daily and weekly sales for each 'family' category and can help identify any seasonality or other patterns within the dataset. 
   - Observations: The daily line plot does not provide clear or significant insights. The weekly line plot shows a general increase in food sales over the years.

*Correlation Analysis*: <br>
1. ScatterPlot
   - The code uses scatterplot to explore the relationship between weekly sales and promotion status of items ('onpromotion') for different 'family' categories in the dataset. A grid of 3x3 subplots is created and Seaborn's regplot function is used to create scatter plots of weekly sales versus onpromotion count as well as a separate plot for the 'SEAFOOD' category. The correlation coefficient is calculated using pandas' corr function on the resampled data, and the best fit line and correlation coefficient are displayed on each plot. The results provide insights into the correlation between weekly sales and promotion status for different 'family' categories.
   - Observation: There is a moderate to high correlation for each category, meaning the variable "onpromotion" has an effect on food sales.
2. BoxPlot
   - The provided code uses boxplot for weekly sales and holiday status of items ('holiday') for different 'family' categories in the dataset. For weekly sales versus holiday count, Seaborn's boxplot is used to create visualizations with correlation coefficients and subplots for each 'family' category. The results provide insights into the relationship between weekly sales and holiday status for different 'family' categories.
   - Observation: There is a high holiday occurrences are associated with higher sales, despite the low correlation.

***SUMMARY***: In general, it appears that food sales have increased over the years, so we should expect current prices to be slightly higher than before. Two variables that we should pay attention to are 'onpromotion' and 'holiday', as these represent the number of items on promotion and the number of holidays occurring in a given week. These variables appear to be factors that influence food sales.

### 3. Model: **ARIMA**
***NOTE 1***: Our strategy for addressing this problem is to train the model to focus exclusively on a single food category. This approach should help the model generalize better than if we were to use data that includes multiple food categories. For the purposes of our example, we have selected the 'BREAD/BAKERY' category. <br>
***NOTE 2***: Since the time series plot is difficult to interpret with so much data from *daily*, we will be converting the time axis to weeks and aggregating the daily sales data into weekly means to provide a clearer picture of the sales trends over time and aid in data exploration and visualization. <br>

*Steps*:
1. I-Value (Stationarity of Data)
   - We used the KPSS test to check the stationarity of the data and found that the test statistic was greater than the critical value at all confidence intervals. This suggests that the time series is non-stationary.
2. Differencing
   - Differencing is a technique commonly used to eliminate trends from non-stationary time series data. After applying differencing to the data, we observed that the trend component of the time series had been successfully removed.
3. MA Value
   - By examining the autocorrelation function (ACF), we can determine the optimal value for the moving average (MA) parameter. We found that the highest correlation occurs at lag 1, indicating that the optimal value for MA is 1.
4. AR Value
   - By analyzing the partial autocorrelation function (PACF), we can determine the optimal value for the autoregressive (AR) component. We found that the highest correlation occurs at lag 1, suggesting that the optimal value for AR is 1.
5. Prediction
   - We train the model using the optimal values for the AR, I, and MA components.
6. Exog Variables
   - To improve the accuracy of our ARIMA model, we included two additional variables, 'onpromotion' and 'holiday', in the model. After retraining the model with these variables, we observed a slight improvement in the model's performance.

### 4. Insights & Conclusion
In this notebook, we managed to train a time series model (ARIMA) to predict food sales for specific categories on a weekly basis has the potential to help businesses better plan their inventory and reduce food waste by avoiding over-ordering. This can contribute to achieving Goal 12 of responsible consumption and production.

Through exploratory data analysis (EDA), we were able to identify key trends and variables that impact food sales, such as the number of items on promotion and the occurrence of holidays. By incorporating these variables as exogenous inputs into our ARIMA model, we were able to improve its accuracy and reliability.

Overall, our approach demonstrates the potential benefits of using data-driven insights and machine learning techniques to optimize business operations and improve sales forecasting. By doing so, businesses can make more informed decisions and ultimately contribute to sustainable and responsible consumption and production practices.

### 5. Contributions
* Ni Jun Hong Glenn
  * Everything
* See Qin Yuan
  * Nothing

### 6. References
