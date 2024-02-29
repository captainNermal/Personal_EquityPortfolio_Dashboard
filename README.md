# Personal_Portfolio_Tool_LSTM
##### Joshua Smith 
## Project Description and Rationale &#x1F4D3;
Personal investing at the retail level can be a hefty and costly endevour:
<p align="center">
  <b>"While these tools [personal analytic tools] can provide valuable insights, their cost can be prohibitive for individual investors. It’s essential to weigh the benefits against the expenses and explore more cost-effective alternatives (Hemming, 2023)". - Analyst Answers</b>
</p>

Personal analytic tools based off publically available information can be incredibly costly and are often not a choice for the average student or retail investor. Once factoring in small incremental postive portfoiolio gains with the fees for the respective software and trade fees,  the economics simply do not make sense. This is a costless tool meant to fill that gap and assist this issue.

Under  the assumption that a respective investor assigns equal weightings to all holdings; this is meant to be a free personal portfolio tool to show up to 5 year historical performance and suggestive forward or implied future performance

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

* Cummulative weighted performance returns
* Chart depiciton of portfolio returns over time
* cummulative beta

<p align="left">
  <b>Forward Metrics: </b>
</p>

* CAPM (Capital Asset Pricing Model) individual and cummulative returns Annualized: $$\[E(R_{\text{asset}}) = R_f + \beta \times (E(R_{\text{market}}) - R_f)\]$$
* Forward Projected Sharpe Ratio Annualized
* Forward implied Volatility for the next day of trading via $Black-Sholes$ call options of underlying equity: $$\[C = S_0 N(d_1) - X e^{-rt} N(d_2)\]$$

Black sholes is a closed form solution (if you have prices in market you can solve for vol). However it is a $non-linear$ equation so a root finding method such as $newton-raphson$ is required to solve for volility denoted as x in $f(x)-0$. Newtons method is iterative such that it uses the derivative of a function to appoximate the root by improving the estimate as the number of simulations go on.

Imagine a simple line chart with x on x axis (vol) and f(x) on y axis with a linear line running through the $x-axis$. Take initial guess $Xn$ (initial vol - usually 0.3 which will be the intersection point between linear line and x axis) then you want to interatively solve for Xn+1 by using equation of linear line $y=mx+b$ to adjust the slope of the linear line to keep improving guess by minimizing or brining $f(x)$ to zero until you converge on the solution.

Slope formula: $\(\frac{{y_2 - y_1}}{{x_2 - x_1}} = m\)$, also known as gradient.

Slope can be translated to: $\(\frac{{0 - f(x)}}{{X_{n+1} - X_n}} = f'(x)\)$

Newton-Raphson method: $\(X_{n+1} = X_n - \frac{{f(X_n)}}{{f'(X_n)}}\)$, iteratively approximates roots.

Application in finance: $\(X_{n+1} = X_n - \frac{{f(X_n) - C_m}}{{\text{vega}}}\)$.

Objective: Minimize $\(f(x) = \text{BS(old vol)} - C_m\)$ to match market price.

Iterative improvement: Replace old info with new info and minimize $\(f(x)\)$ through break conditions.
