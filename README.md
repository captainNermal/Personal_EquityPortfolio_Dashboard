# Personal_EquityPortfolio_Dashboard
##### Joshua Smith 
## Project Description and Rationale &#x1F4D3;
Personal investing at the retail level can be a hefty and costly endevour:
<p align="center">
  <b>"While these tools [personal analytic tools] can provide valuable insights, their cost can be prohibitive for individual investors. It’s essential to weigh the benefits against the expenses and explore more cost-effective alternatives (Hemming, 2023)". - Analyst Answers</b>
</p>

Personal analytic tools based off publically available information can be incredibly costly and are often not a choice for the average student or retail investor. Once factoring in small incremental postive portfoiolio gains with the fees for the respective software and trade fees,  the economics simply do not make sense. This is a costless tool meant to fill that gap and assist this issue.

Under  the assumption that a respective investor assigns equal weightings to all holdings; this is meant to be a free personal portfolio tool to show up to 5 year historical performance and suggestive forward or implied future performance

## A note on Nans or Empties &#x2753;

Nans are a factor that everyone has to deal with. In this case we shall deal with hthem dynamically. By observing the total length of a dataframe and then the number of rows that contain Nans, one can set up a dynamic interpretation. You must be very careful withj missing data as you have many options: you can drop values, perform mean or linear interprolations among many more methods. Note that one cannot simply drop Nans if they are a large component of the data (you could miss critical values in other columns across the name row); thus, you may have to simply "roll" with Nans in this case. Note that mean or linear interprolations can also present an equally pressing issue if Nans maker up a large component of the data, having the mean or linear interprolation of missing points (porvided they compose a large amount of the data could drastically skew the distribution, accuracy and results. Thus, dynamic interpolations are required:

* If Nans make up less than 2% of the data, the Nans and thier respective rows are $dropped$
* If Nans are between 2% <= x <= 10% of the data, then the Nans are interprolated via $linear interprolation$
* If Nans are greater than 10% of the data, we can utalize $spline interprolation$ (overlapping polynomial function curves to interpret missing data). This is logical as splines are capable of adjsuting the the often non-lionear patterns in financial data 


## What Metrics Will You Be provided With &#x2753;
<p align="left">
  <b>Historical Metrics: </b>
</p>

* Current risk free rate denoted by the $10-Y$ US Government Daily Treasury Par Real Yield Curve Rate (Often utalzied as a discount factor for public assets or products)
* A correlation matrix with $correlation-coefficient$: $$-1 < r < 1$$ To observe relationship and strength of the relationship regarding the interaction between portfolio holdings and the greater market with the S&P500 Index serving as a $market-proxy$
* Individual 5 year $Mins$ for each holding
* Individual 5 year $Maxs$ for each holding
* Individual 5 year $Mean$ for each holding  and Annualized $Mean$
* Individual 5 year $Std-Dev$ or $volitility$ for each holding and annualized $Std-Dev$
* Individual 5 year $Sharpe-ratio$ and Annualized $Sharpe-ratio$ (a metric used to show relative risk to reward) $$\[\text{Sharpe Ratio} = \frac{R_i - R_f}{\sigma_i}\]$$
* Individual 5 year $Beta$ or $\( \beta \)$ for each holding (A holdings risk relative to the market that possesses a $\( \beta \)=1$)
  Beta is calculated via slope method:
  $$\beta = \frac{Cov(R_i, R_m)}{Var(R_m)}$$

* Cummulative weighted performance returns (Log Normal Returns as an adjustment for typically very right skewed financial data as we have high concentrations of small postive returns and a few high positve returns, by taking teh natural log our distribution becomes more symetrical as we condense high returns and expand small returns
* Chart depiciton of portfolio returns over time
* cummulative beta

<p align="left">
  <b>Forward Metrics: </b>
</p>

* CAPM (Capital Asset Pricing Model) individual and cummulative returns Annualized: $$\[E(R_{\text{asset}}) = R_f + \beta \times (E(R_{\text{market}}) - R_f)\]$$
* Forward Projected Sharpe Ratio Annualized
* Forward implied Volatility for the next 6 months via $Black-Sholes$ call options of underlying equity: $$\[C = S_0 N(d_1) - X e^{-rt} N(d_2)\]$$

Black sholes is a closed form solution (if you have prices in market you can solve for vol). However it is a $non-linear$ equation so a root finding method such as $newton-raphson$ is required to solve for volility denoted as x in $f(x)-0$. Newtons method is iterative such that it uses the derivative of a function to appoximate the root by improving the estimate as the number of simulations go on.

Imagine a simple line chart with x on x axis (vol) and f(x) on y axis with a linear line running through the $x-axis$. Take initial guess $Xn$ (initial vol - usually 0.3 which will be the intersection point between linear line and x axis) then you want to interatively solve for Xn+1 (new volitilty) by using equation of linear line $y=mx+b$ to adjust the slope of the linear line to keep improving guess by minimizing or brining $f(x)$ to zero until you converge on the solution.

Slope formula: $\(\frac{{y_2 - y_1}}{{x_2 - x_1}} = m\)$, also known as gradient.

Slope can be translated to: $\(\frac{{0 - f(x)}}{{X_{n+1} - X_n}} = f'(x)\)$

Newton-Raphson method: $\(X_{n+1} = X_n - \frac{{f(X_n)}}{{f'(X_n)}}\)$, iteratively approximates roots.

Application in finance: $\(X_{n+1} = X_n - \frac{{BS(old vol) - C_m}}{{\text{vega}}}\)$.

Objective: Minimize $f(x)$ to match market price.

Iterative improvement: iteratively replace ${BS(old vol)}$ with new volitilty or $X_{n+1}$

$Note*$ we are solving for 6 month implied volitilty, however we are using the call option contract that expires closest to $today$ and adjusting the tiime factor in the equation for 6 months $T=0.5$. This is done for two reasons. The method in which the data is structure on Yahoo Finance and account of a liquidity arguement. Yahoo Finance allows you to select a drop down menu of expiry dates; however these dates will never quite resemble 6 months (It could be 4.5, 7 or so on). Thus by reducing to a daily increment and adjusting T=0.5 for 6 months promotes simplicity. Looking to the second arguement, maximum liquidity is achieved by the closest option contract to today reflecting the most non-distorted option price. If we look 6 months out, depending on the holding, liquidity can potentially "dry up" leading to a $wider-bid/ask-spread$ and distorting the option price, hence affecting the input to the implied volitilty calculation. By taking the most liquid price and adjusting for 6 months - the most accurate results are rendered. 

<p align="left">
  <b>Advanced: </b>
</p>

* Predictive time series

We shall be using $RNN$, more specifically, $LSTM$ to do a 30 day predictive portfolio. RNNs are great for sequential dependencies. Hence new data points depend very heavily on previous data points. As a result, a predictive time series utilizing RNNs would be incredibly appropriate for financial data. Oftentimes markets are very short-term reactionary, meaning that when there is a couple day bullish or bearish momentum the psychological aspect of the market will usually dictate that the next day pricing is impacted by this occurrence. However, the gradient on RNN models can become exponentially large and small very quickly and frankly in “blow up”. LSTM is a subset of RNN that can avoid this gradient issue.

<p align="center">
  <b>It is a common known fact that using predictve time series for financial markets is not advisable as markets are incredibly complex. None the less, lets give it a try...</b>
</p>

* Suggested SMA strategy (Standard Moving Average)
* Suggested EMA startegy (Exponential Moving Average)

## GUIs, Websites, and Libraries &#x1F4F6;
Libraries and Websites: 

* Numpy: used for statistical data manipulation

* Datetime: to assist with date interaction and manipulation

* Pandas: used for statistical data manipulation

* yfinance: utalized to extract data from websitees and online sources via scraping

* Pandas_makret_calendars: utalized to assist in finding the closest trading dates exempt of holidays and market shutdowns to aid in scraping for pricing data

* Matplot: used for statistical data manipulation and graphical respresentation

* urllib.request: utalized to get URL quests to help obtain URL tags for webscraping of live (slight time delay) prices and rf rate

* beautiful soup: library used for webscraping and webscraping features/ function

* scripy: to assist with computation and CDF funcvtionality to convert from zscore or other metric to probabiltiy 

* math: to aid in computation with certain terms, variables or metrics such as 'e' for exponential components of calcuations (Black Sholes, etc)

* sklearn: to provide scaling assistance with the input data for LSTM

* keras from tensorflow: to assist with implementation of LSTM ML concepts for forward portfolio forcasting

Website(s):

* Yahoo Finance: utalized as a proxy for a conventional platform often found within financial institutions such as bloomberg terminal or Capital IQ

GUI Description: Tkinter

* Advantages: cross platform functionality, rapid development, easy to use
* Disadvantages: default look, limited widgets and doesnt support threading


## How to Use Model &#x2611;

1. Simply run the program.
2. A dialogue box will appear, enter your publically traded assets and the exchange they trade on
3. Click "Go"
4. Close the dialogue box and let the program run
5. a tkinter window will pop up - you are capablke of switching pages



