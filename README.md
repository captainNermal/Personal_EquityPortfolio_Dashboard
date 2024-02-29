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
