# Credit Card Default Prediction
Classification problem to predict if a customer will default on next month's credit card payment using the [UCI dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

Access the [streamlit](https://fds-project-cmse830-credit-card-default-prediction-r2f68jop9pe.streamlit.app/) web-app to view the insights drawn from the dataset.

## Table of Contents:
1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
4. [Exploratory Data Analysis](https://fds-project-cmse830-credit-card-default-prediction-r2f68jop9pe.streamlit.app/)
5. [Future Work](#future-work)

## Introduction:
Credit Risk is the inability of the receiver to pay back the loan at the designated time which was decided by the lender and the borrower during the loan agreement. This causes major concerns among the financial institutes as it can result in “credit defaulting”, which can prove to be drastic to the lending party, as it may lead to losses and even bankruptcy. A thorough evaluation and verification of the ability of a borrower to repay their loan in the decided time period can result in minimized credit risk and prove beneficial for financial institutes.

Machine Learning models can be deployed to predict risky customers and minimise lenders' losses. By using algorithms to study the behaviour and demographics of previous customers, we can apply the findings to customers in the future and differentiate between risky and non-risky customers, resulting in efficient loan lending.

## Datasets:
**I. Default of Credit Card Clients**

Dataset contains information on credit card clients in Taiwan from April 2005 to September 2005. It has 30,000 instances across 25 attributes, contains multivariate characteristics, and the attributes have both integer, categorical and real data types. The attribute summary is as follows:

`ID`: ID of each client

`LIMIT_BAL`: Amount of given credit in NT dollars (includes individual and family/supplementary credit)

`SEX`: Gender (male, female)

`EDUCATION`: Level of education (graduate school, university, high school, others)

`MARRIAGE`: Marital status (married, single, others)

`AGE`: Age in years

`PAY_0`: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)

`PAY_2`: Repayment status in August, 2005 (scale same as above)

`PAY_3`: Repayment status in July, 2005 (scale same as above)

`PAY_4`: Repayment status in June, 2005 (scale same as above)

`PAY_5`: Repayment status in May, 2005 (scale same as above)

`PAY_6`: Repayment status in April, 2005 (scale same as above)

`BILL_AMT1`: Amount of bill statement in September, 2005 (NT dollar)

`BILL_AMT2`: Amount of bill statement in August, 2005 (NT dollar)

`BILL_AMT3`: Amount of bill statement in July, 2005 (NT dollar)

`BILL_AMT4`: Amount of bill statement in June, 2005 (NT dollar)

`BILL_AMT5`: Amount of bill statement in May, 2005 (NT dollar)

`BILL_AMT6`: Amount of bill statement in April, 2005 (NT dollar)

`PAY_AMT1`: Amount of previous payment in September, 2005 (NT dollar)

`PAY_AMT2`: Amount of previous payment in August, 2005 (NT dollar)

`PAY_AMT3`: Amount of previous payment in July, 2005 (NT dollar)

`PAY_AMT4`: Amount of previous payment in June, 2005 (NT dollar)

`PAY_AMT5`: Amount of previous payment in May, 2005 (NT dollar)

`PAY_AMT6`: Amount of previous payment in April, 2005 (NT dollar)

`default payment next month`: Default payment (yes, no)

**II. Macroeconomic Data**

Labour, income and inflation data for the selected period in Taiwan has been taken from [National Statistics Republic of China (Taiwan)](https://eng.stat.gov.tw/cl.aspx?n=2324) and [DGBAS Government Bureau](https://www.dgbas.gov.tw/default.aspx).

`CPI`: Consumer Price Index representing the average change over time in the prices paid by consumers for a representative basket of consumer goods and services

`Unemployment Rate`: Percentage of people in the labour force who are unemployed (includes civilians age 15 & above who were： (i) jobless (ii) available for work (iii) seeking a job or waiting for results after job seeking during the reference week (iv) waiting for a recall after layoff (v) having a job offer but have not started to work)

`Avg Income Level`: Disposable income of employees (including those having: (i) full-time, part-time, or another payroll (ii) entrepreneurial income (iii) property income (iv) imputed rent income (v) current transfer receipts)

## Data Cleaning and Preprocessing:


## Future Work:
