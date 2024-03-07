"""
Name: Smith Joshua
Date: April, 2023
Project: Time Series - Personal Portfolio Tool
"""

"""
Description: Personal Portfolio tool to show historical performance of a proposed portfolio then predicted forward thinking performance metrics such as sharpe ratio, implied
volitility, predictive time series 
    - utalize S&P500 as proxy for the market
    -> To view more look to read me file 

"""

"""
GUI Description: Tkinter
    -> Advantages: Cross platform functionality, rapid development, easy to use
    -> default look, limited widgets and doesnt support threading
    -> construction: utalize a class object to predefine functionality as in a class we have everything pre structured
"""

"""
External libraries and Sources used: A variety of libraries and external websites shall be used to manipuate and obtain/ scrape data to avoid the simplicity of File I/O
    Libraries:
        -> Numpy: used for statistical data manipulation
        -> Datetime: to assist with date interaction and manipulation
        -> Pandas: used for statistical data manipulation
        -> yfinance: utalized to extract data from websitees and online sources via scraping
        -> Pandas_makret_calendars: utalized to assist in finding the closest trading dates exempt of holidays and market shutdowns to aid in scraping for pricing data
            -o: https://www.learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/
        -> Matplot: used for statistical data manipulation and graphical respresentation
        -> urllib.request: utalized to get URL quests to help obtain URL tags for webscraping of live (slight time delay) prices and rf rate
        -> beautiful soup: library used for webscraping and webscraping features/ function
        -> py_vollib.black_scholes & py_vollib.black_scholes.greeks.analytical: to assist in iteritive simulations to solve for implied volitility and access greeks 
        -> scripy: to assist with computation and CDF funcvtionality to convert from zscore or other metric to probabiltiy 
        -> math: to aid in computation with certain terms, variables or metrics such as 'e' for exponential components of calcuations (Black Sholes, etc)
        -> statsmodels: to aid in decomposing time series into seasonality, trend and noise 
        -> sklearn: to provide scaling assistance with the input data for LSTM
        -> keras from tensorflow: to assist with implementation of LSTM ML concepts for forward portfolio forcasting
    Website(s):
        -> Yahoo Finance: utalized as a proxy for a conventional platform often found within financial institutions such as bloomberg terminal or Capital IQ
"""



#library implementation 
import numpy as np 
import pandas as pd
import datetime
import pandas_market_calendars as mcal
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import tkinter as tk
from tkinter import ttk
from scipy.stats import norm
import math
#library installation - concerning time series analysis in forward projection component of code
from statsmodels.tsa.seasonal import seasonal_decompose 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam



#Tkinter GUI input construction 
class openGUI:
    #establish entry interface
    def __init__(self):
        #initialize GUI
        self.root = tk.Tk()
            #label number one
        self.label = tk.Label(self.root, text="Enter tickers seperated by commas: ")
        self.label.pack() #pack to implement
            #entry number one initialization for user_input(self) function within openGUI class
        self.entry = tk.Entry(self.root)
        self.entry.pack()
            #label number two
        self.label_two = tk.Label(self.root, text="Enter the exchange: ")
        self.label_two.pack()
            #entry number two
        self.entry_two = tk.Entry(self.root)
        self.entry_two.pack()
            #runbutton
        self.run_button = tk.Button(self.root, text="Run", command=self.user_input) 
        self.run_button.pack()

    def user_input(self):
        #collect entry one - use self to define list as variable to interact with in main variable
        user_Input_collection = self.entry.get()
        self.user_Input_collection_list = list(user_Input_collection.split(",")) 
        #collect entry two
        user_Input_collection_two = self.entry_two.get()
        self.user_Input_collection_list_two = user_Input_collection_two 



#Tkinter GUI output construction 
class display_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Simulation Summary")
        # Main container frame
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.main_frame['background'] = f'#{153:02x}{204:02x}{255:02x}'
        # Historical and Projection frames
        self.historical_frame = tk.LabelFrame(self.main_frame, text="Historical Metrics")
        self.projection_frame = tk.LabelFrame(self.main_frame, text="Forward Projection Metrics")

        self.create_historical_page(None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None) 
        self.create_projection_page(None,None,None,None,[],[],[],[],[],[],None,None) # cant have none for list of graph data so replace with empty lsit
    
        self.show_historical_page()

    #create historical page 
    def create_historical_page(self,ten_Year_rfrate,mean_Dict,stddev_Dict,sharpe_Dict,mean_Dict_annual,stddev_Dict_annual,min_Val_cummulative,max_Val_cummulative,mean_Val_cummulative,
                               stddev_Val_cummulative,stddev_Val_cummulative_annualized,sharpe_Ratio_cummulative_annualized,beta_Dict,weighted_beta,historical_Dataframe_correlation_matrix,mean_Val_cummulative_annualized):
        
        # Add historical information widgets here
        labels = [
            f'Ten year risk free treasury rate(%): {ten_Year_rfrate}',
            f'5 Year Daily standard mean per holding(%): {mean_Dict}',
            f'5 Year Daily standard deviation per holding: {stddev_Dict}',
            f'5 Year Annualized Sharpe Ratio per holding: {sharpe_Dict}',
            f'5 Year Annual standard mean per holding(%): {mean_Dict_annual}',
            f'5 Year Annual standard deviation per holding: {stddev_Dict_annual}',
            f'5 Year Daily Cumulative performance (Log Normal) minimum(%): {min_Val_cummulative}',
            f'5 Year Daily Cumulative performance (Log Normal) maximum(%): {max_Val_cummulative}',
            f'5 Year Daily Cumulative performance (Log Normal) standard mean(%): {mean_Val_cummulative}',
            f'5 Year Annual Cumulative performance (Log Normal) standard mean(%): {mean_Val_cummulative_annualized}',
            f'5 Year Daily Cumulative performance (Log Normal) standard deviation: {stddev_Val_cummulative}',
            f'5 Year Annual Cumulative performance (Log Normal) standard deviation: {stddev_Val_cummulative_annualized}',
            f'5 Year Annual Cumulative performance (Log Normal) Sharpe ratio: {sharpe_Ratio_cummulative_annualized}',
            f'5 Year Rolling Beta per holding: {beta_Dict}',
            f'5 Year Rolling Beta Cumulative Holding: {weighted_beta}',
            f'5 Year correlation matrix: \n {historical_Dataframe_correlation_matrix}'
        ]

        for idx, label_text in enumerate(labels, start=1):
            label = tk.Label(self.historical_frame, text=label_text, wraplength=400, justify="left", font=('Arial',8))
            label.grid(row=idx, column=0, sticky='w', padx=10, pady=5)

        # Button to switch to Projection page
        historical_button = tk.Button(self.historical_frame, text="Forward Projection Metrics", command=self.show_projection_page)
        historical_button.grid(row=len(labels)+1, column=0, padx=10, pady=5)
        
    #create projection page
    def create_projection_page(self,implied_Vol_dict,capm_Dict_individual_dict,forward_Sharpe_ratio_dict,strat_position_dict,historical_backtest_on_test_data,future_predictions,sma_list,ema_list,train_Predictions,train_Data_y_adjust_1d,train_loss,train_root_mse):
        
        # Add forward projection widgets here
        labels = [
            f'Forward 6 Month Implied Volatility per holding: {implied_Vol_dict}',
            f'CAPM Annual Expected Return per holding(%): {capm_Dict_individual_dict}',
            f'Forward Annual Sharpe ratio per holding: {forward_Sharpe_ratio_dict}',
            f'Forward SMA Portfolio strategy(%): {strat_position_dict}',
            #f'Model Training Loss is {train_loss} and Model Training root MSE is {train_root_mse}'
        ]

        for idx, label_text in enumerate(labels, start=1):
            label = tk.Label(self.projection_frame, text=label_text, wraplength=400, justify="left", font=('Arial',7)) # 400
            label.grid(row=idx, column=0, sticky='w', padx=10, pady=5) #5

        self.create_matplotlib_graph(self.projection_frame,historical_backtest_on_test_data,future_predictions,sma_list,ema_list,train_Predictions,train_Data_y_adjust_1d)

        # Button to switch to Historical page
        projection_button = tk.Button(self.projection_frame, text="Historical Metrics", command=self.show_historical_page)
        projection_button.grid(row=len(labels)+1, column=0, padx=10, pady=5) 



    #create matplot graphs 
    def create_matplotlib_graph(self,frame,historical_backtest_on_test_data,future_predictions,sma_list,ema_list,train_Predictions,train_Data_y_adjust_1d):
        
        #create graph one
        fig = Figure(figsize=(6,2), dpi=90)#6
        plot = fig.add_subplot(1, 1, 1)
        plot.plot(train_Predictions, label='Test Predictions')
        plot.plot(train_Data_y_adjust_1d,                            label='Actual Values')
        plot.set_title('Historical Backtest on Test Data - Last 835 Days',fontsize=9)
        plot.set_ylabel('Return (%)', fontsize=8)
        plot.set_xlabel('Time duration in Days (D)', fontsize=8)
        plot.legend()
        plot.tick_params(axis='x', labelsize=6)
        plot.tick_params(axis='y', labelsize=6)

        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=6, column=0, padx=10, pady=5)


        #create graph two
        fig2 = Figure(figsize=(6,2), dpi=90)
        plot2 = fig2.add_subplot(1, 1, 1)
        plot2.plot(future_predictions)
        plot2.set_title('Future Portfolio Return Predict. - Next 30 Days',fontsize=9)
        plot2.set_ylabel('Return (%)', fontsize=8)
        plot2.set_xlabel('Time duration in Days (D)', fontsize=8)
        plot2.tick_params(axis='x', labelsize=6)
        plot2.tick_params(axis='y', labelsize=6)

        
        canvas2 = FigureCanvasTkAgg(fig2, master=frame)
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=7, column=0, padx=10, pady=5)


        #create graph three
        fig3 = Figure(figsize=(6,2), dpi=90)
        plot3 = fig3.add_subplot(1, 1, 1)
        plot3.plot(future_predictions, label='Future Predictions')
        plot3.plot(sma_list, label='SMA')
        plot3.plot(ema_list, label='EMA')
        plot3.legend()
        plot3.set_title('Future Portfolio Return Predict. - Next 30 Days - SMA And EMA',fontsize=9)
        plot3.set_ylabel('Return (%)', fontsize=8)
        plot3.set_xlabel('Time duration in Days (D)', fontsize=8)
        plot3.tick_params(axis='x', labelsize=6)
        plot3.tick_params(axis='y', labelsize=6)

        canvas3 = FigureCanvasTkAgg(fig3, master=frame)
        canvas3.draw()
        canvas3.get_tk_widget().grid(row=7, column=0, padx=10, pady=5)
        


    #show historical page
    def show_historical_page(self):

        self.projection_frame.pack_forget()
        self.historical_frame.pack(fill='both', expand=True, padx=10, pady=10)#grid(row=0, column=0, padx=10, pady=10)
        


    #show projection page 
    def show_projection_page(self):
        
        self.historical_frame.pack_forget()
        self.projection_frame.pack(fill='both', expand=True, padx=10, pady=10)#grid(row=0, column=0, padx=10, pady=10)




"""
See Relevant factors below that coenside with traditional historical metrics for a 5 year analytical period assuming equal weightings
- Assumptions: Assume equal portfolio weighting
    - General:
        -o: Current Rf rate
        -o: correlation matrix
    - Individual:
        -o: Min
        -o: Max
        -o: Mean | Mean annualualized
        -o: Std Dev | Std dev annualized (volatility)
        -o: Sharpe ratio annualized
        -o: Beta
    - Cummulative:
        -o: Cummulative weighted performance returns
        -o: Chart depiciton of portfolio returns over time
        -o: cummulative beta
"""



def historical_data_retrieve(user_Input_list, user_Input_list_two):
    
    #define today and 5 years ago # fixed dates
    today_date = datetime.datetime.today()
    date_5_years_ago_from_today = today_date - datetime.timedelta(365*5)

    ###define exchange date retrieval
    #user specified exchange
    user_specified_exchange = f'{user_Input_list_two}'
    
    #retrieve schedule
    exchange_dates = mcal.get_calendar(user_specified_exchange)
    trade_Range_dates = exchange_dates.schedule(start_date=date_5_years_ago_from_today, end_date=today_date)

    closest_To_today = trade_Range_dates.index[-1].date()
    closest_To_date_fiveyearsago = trade_Range_dates.index[0].date()

    ###retrieve historical data and push to empty dataframe
    #establish epmty dataframe
    historical_Dataframe_list = []
    
    #retrieve historicals and push to dataframe
    for i in range(len(user_Input_list)):
        ticker_Index_retrieve_yfformat = yf.Ticker(user_Input_list[i])
        historical_Data_item = ticker_Index_retrieve_yfformat.history(period="1d", start=date_5_years_ago_from_today, end=today_date)
        historical_Data_item_conversion = historical_Data_item["Close"]

        historical_Dataframe_list.append(historical_Data_item_conversion)
    #concat histporical lsit to create dataframe from list
    historical_Dataframe = pd.concat(historical_Dataframe_list, axis=1)
    historical_Dataframe.columns = user_Input_list


    ###convert histoircal dataframe to natural log normal for positive skew and fat tail to better represent financial markets
    #refer to VarModelProject or README file for manual log calculation, here we are subtracting yesterdays price log from todays price log, difference corresponds to todays logorithm
    #ln(x)-ln(y) = ln(x/y) = (Current-Prior)/Prior with natural log smoothing
    historical_Dataframe = (np.log(historical_Dataframe) - np.log(historical_Dataframe.shift(1))) * 100 #*100 to convert to whole percents opposed to decimals
    historical_Dataframe = historical_Dataframe.dropna()

    return historical_Dataframe, user_Input_list, closest_To_today, closest_To_date_fiveyearsago



#rf scraper 
def rf_Rate_scrape():
    
    #link request
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_real_yield_curve&field_tdr_date_value=2023"
    html = urlopen(url)
    soup = BeautifulSoup(html)

    #empty list
    tbill_Rate_list = []

    #extract tbill rates and isolate ten year
    all_Rows = soup.find_all("tr")
    for i in all_Rows:
        row_List = i.find_all("td")
    for i in row_List:
        tbill_Rate_list.append(i.text)

    ten_Year_rfrate = tbill_Rate_list[3]
    ten_Year_rfrate = round(float(ten_Year_rfrate),3)

    return ten_Year_rfrate



#historical stats
def historical_Stats(historical_Dataframe, user_Input_list, ten_Year_rfrate):
    
    #create dictionary key value pairs for key stats
    min_Dict = {}
    max_Dict = {}
    mean_Dict = {}
    mean_Dict_annual = {}
    stddev_Dict = {}
    stddev_Dict_annual ={}
    sharpe_Dict = {}

    #perform calculations
    for i in range(len(user_Input_list)):
        #min and max - 5 year range
        min_Val = historical_Dataframe.iloc[i,0] #use iloc for df
        max_Val = 0
        for j in historical_Dataframe[f'{user_Input_list[i]}']:
            if j > max_Val:
                max_Val = j
        for j in historical_Dataframe[f'{user_Input_list[i]}']:
            if j < min_Val:
                min_Val = j
        max_Dict[f'{user_Input_list[i]}'] = max_Val
        min_Dict[f'{user_Input_list[i]}'] = min_Val

        #mean and std dev (mean not annualized - std deviation is)
        mean_Val = 0
        for k in historical_Dataframe[f'{user_Input_list[i]}']:

            mean_Val = mean_Val + (k / len(historical_Dataframe[f'{user_Input_list[i]}']))
            mean_Dict[f'{user_Input_list[i]}'] = round(mean_Val,3)
        annualized_Geomean_val = ((1 + (mean_Val/100))**(252)-1)*100
        mean_Dict_annual[f'{user_Input_list[i]}'] = round(annualized_Geomean_val,3)

        variance_Sum = 0
        for k in historical_Dataframe[f'{user_Input_list[i]}']:
            variance_Sum = variance_Sum + ((k - mean_Dict[f'{user_Input_list[i]}'])**2)

        stddev_Val = (variance_Sum / len(historical_Dataframe[f'{user_Input_list[i]}']))**(1/2)
        stddev_Dict[f'{user_Input_list[i]}'] = round(stddev_Val,3)

        annualized_Stddev_val = stddev_Val * ((252)**0.5)
        stddev_Dict_annual[f'{user_Input_list[i]}'] = round(annualized_Stddev_val,3)

        #sharpe ratio
        sharpe_Ratio_annualized = (annualized_Geomean_val - ten_Year_rfrate) / annualized_Stddev_val
        sharpe_Dict[f'{user_Input_list[i]}'] = round(sharpe_Ratio_annualized,3)

    return historical_Dataframe, mean_Dict, stddev_Dict, sharpe_Dict, mean_Dict_annual, stddev_Dict_annual



#cummulative historical stats on log normal cummulative performance
def historical_Stats_cummulative(historical_Dataframe, user_Input_list, ten_Year_rfrate):
    
    #assume equal weighting for historical performance
    historical_Dataframe["Cummulative Performance"] = historical_Dataframe.loc[:, historical_Dataframe.columns != '^GSPC'].sum(axis=1)/ (len(user_Input_list)-1)

    #min and max calcuation
    min_Val_cummulative = historical_Dataframe.iloc[-1,0]
    max_Val_cummulative = 0
    for i in historical_Dataframe["Cummulative Performance"]:
        if i > max_Val_cummulative:
            max_Val_cummulative = round(i,3)
    for j in historical_Dataframe["Cummulative Performance"]:
        if j < min_Val_cummulative:
            min_Val_cummulative = round(j,3)

    #mean calculation
    mean_Val_cummulative_numerator = 0
    for i in historical_Dataframe["Cummulative Performance"]:
        mean_Val_cummulative_numerator += i
    mean_Val_cummulative = round(mean_Val_cummulative_numerator/ len(historical_Dataframe["Cummulative Performance"]),3)
    mean_Val_cummulative_annualized = round(((1+(mean_Val_cummulative/100))**(252)-1) * 100,3)

    #varaicne and standard deviaton
    variance_Sum_cumulative = 0
    for i in historical_Dataframe["Cummulative Performance"]:
        variance_Sum_cumulative = variance_Sum_cumulative + ((i - mean_Val_cummulative)**2)
    stddev_Val_cummulative = round((variance_Sum_cumulative / len(historical_Dataframe["Cummulative Performance"]))**(1/2),3)
    stddev_Val_cummulative_annualized = round(stddev_Val_cummulative * ((252)**(1/2)),3)

    #sharpe ratio
    sharpe_Ratio_cummulative_annualized = round((mean_Val_cummulative_annualized - ten_Year_rfrate) / stddev_Val_cummulative_annualized,3)

    return min_Val_cummulative, max_Val_cummulative, mean_Val_cummulative, mean_Val_cummulative_annualized, stddev_Val_cummulative, stddev_Val_cummulative_annualized, sharpe_Ratio_cummulative_annualized



# calcaute beta
def beta_calc(historical_Dataframe, user_Input_list, mean_Dict, stddev_Dict):
    
    #alter dataframe to get correct data
    historical_Dataframe.drop("Cummulative Performance", axis=1, inplace=True)

    #calculate individual beta(s) via slope method
    beta_Dict = {}

    for i in range(len(user_Input_list)):
        # covariance (numerator) between market proxy and specific holding --> (Sum(x - xbar)(y - ybar)) / n-1 --> y is market
        sum_Of_numerator_product = 0
        for j in range(len(historical_Dataframe[f'{user_Input_list[i]}'])):
            sum_Of_numerator_product = sum_Of_numerator_product + (((historical_Dataframe[f'{user_Input_list[i]}'][j]) - (mean_Dict[f'{user_Input_list[i]}']))*((historical_Dataframe["^GSPC"][j]) - (mean_Dict[f'^GSPC'])))

        covariance = sum_Of_numerator_product / (len(f'{historical_Dataframe[user_Input_list[i]]}') -1)

        beta_Val = covariance / ((stddev_Dict[f'^GSPC'])**2) #raise to power of 2 to convert standard deviation to variance
        beta_Dict[f'{user_Input_list[i]}'] = round(beta_Val,3)

    #Remove market proxy beta from dict
    beta_Dict.pop('^GSPC')

    #calculate cummulative beta via weighted beta method
    weighted_beta = 0
    for i in beta_Dict:
        weighted_beta = weighted_beta + (beta_Dict[i] / (len(beta_Dict)))
    weighted_beta = round(weighted_beta,3)

    return beta_Dict, weighted_beta



#correlatioin matrix
def correlation_Matrix(historical_Dataframe, user_Input_list):
    
    # we could simply use pandass correlation matrix functionality with df.corr() as it is optimized, well tested, computatinally efficient, and stadnard practice
    # to do so manually, although possible would be inneficient, time intensive provided I am not modifying standard process
    # manual formula is covariance(x,y) / Product (std(x), std(y)) --> see manual calcualtion above for covariance where we subract mean --> Sum(x - xbar)(y - ybar)) / product(std(x)*std(y)) as we are normailziing to get correlation matrix
    historical_Dataframe_correlation_matrix = historical_Dataframe.corr()

    return historical_Dataframe_correlation_matrix



"""
Forward Projecting Metrics
- Assumptions: Assume equal portfolio weighting
    - General
        -o: CAPM Expected return B(Mrkt - rf) + rf | individual and cummulative
        -o: Forward Projecting Sharpe ratio | individual and cummulative
        -o: Forward implied volitility | individual and cummulative
            -i: ideally you want the maturity or expiry of the options contract tht you obtain K and market price of option from to match the time factor (T)
                of the implied volitlity you re looking to calculate (i.e. you would obtain K and last market price from a call option contract with an expirey - many ways to do it and all are correct as long as you can justify -
                6 months from now if you wanted to calculate implied volatilty for the next 6 months (T=0.5)). This is for optimal accuracy. However, provided both the structure
                of the options data from yahoo finance and yfinance(autoloadclosest, aceess scrolly and load new page for innacure 4.5 or 7 months out so not even 6 motnhs really) coupled with the desire to mimick a base case in which liquidty is at its highest point (resulting in a narrow bid ask spread making a more accurate input to BS model - of course downside this is maybe market is pricing something in to contract 6 motnhs out tht we dont accoutn for)for simplicity, we shall use the option with a
                last trade date (last date in which the option is available to trade prior to expirey) that is closest to today and then we adjust t in equaltion to 0.5 years or 6 months
            -i: Black Sholes formula is a closed form solution (if you have prices in market you can solve for vol). To do this you need to use root finding method (Newton) to find solution to 
                equation. solving for x in f(x) - 0. Newtons method is iterative such that it uses the derivative of a function to appoximate the root by improving teh estimate as the number of simulations go on.
                Imagine a line graph with x on x axis and f(x) on y axis. Take initial guess Xn then you want to solve for Xn+1 by using equation of linear line y=mx+b to keep improving guess until you converge on the solution. Xn initial guess is intersection on x axis and linear line and y axis is f(x) so just change slope.
                    - y2 - y1 / x2 - x1 = m --> also known as slope
                    - 0 - f(x) / Xn+1 - Xn = f'(x)
                    - Xn+1 = Xn - (f(Xn) / f'(Xn)) --> converge on solution by iterively approximating Xn+1
                    - Xn+1 = Xn - (f(Xn) / f'(Xn)) == vol(new) = BS(old vol) - Cm / vega 
                    - we want to converge or minimize equation f(x) = BS(old vol) - Cm by matching the BS old vol market rpice and current market price
                    -improving guess by replacing old info with new info and then minimizing F(x) through series of if break conditions            
    - Advanced
        -o: Predictive time series
        -o: Suggested SMA strategy
        -o: Suggested EMA startegy

"""



#idealliy you want to use py_vollib library, however the library experienced scalar issues when running though the vega variable, computation although longer manually ran without scalar issues in the gamma variable
def black_Sholes_newton_impliedvol(user_Input_list, ten_Year_rfrate, closest_To_today, closest_To_date_fiveyearsago):

    #establish dicrionaries for varaibles
    last_Market_price_S0_dict = {} # S0 --> last price of underlying asset in market
    strike_Price_K_dict = {} # K --> the strike price on the option contract
    option_Market_price_dict = {} # Price of the option in the market

    #alter user input list to remove the s&p market proxy ^GSPC
    user_Input_list.pop(-1)

    """
    obtain S0,K, and market price option and stock data with a maturty a year from now: better demonstrate long term expectations, but at same time you might not have much liquidity a year out leading to wider bid ask spreads
    resulting in a skewed last option market price affecting the implied volatility metric
    """
    for i in range(len(user_Input_list)):
        index_ticker_retrieve = yf.Ticker(user_Input_list[i])

        #last price S0
        historical_Data_last_price = index_ticker_retrieve.history(period="1d", start=closest_To_date_fiveyearsago, end=closest_To_today)
        historical_Data_last_price_close = historical_Data_last_price["Close"][-1]
        last_Market_price_S0_dict[f'{user_Input_list[i]}'] = historical_Data_last_price_close

        ###strike price K, option price
        #filter and clean yfinance option daara         
        historical_Data_options_maturity_dates = index_ticker_retrieve.option_chain() #retrieve maturity or settlement lsit dates
        historical_Data_options_maturity_dates_call_lasttradedates = list(historical_Data_options_maturity_dates.calls['lastTradeDate']) # return only the trade dates IN timestamp
        historical_Data_options_maturity_dates_call_lasttradedates_string = [str(x) for x in historical_Data_options_maturity_dates_call_lasttradedates] #convert timestamp to string 
        historical_Data_options_maturity_dates_call_lasttradedates_datetime = [datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z') for date_str in historical_Data_options_maturity_dates_call_lasttradedates_string] #convert string to datetime UTC
 
        static_Date_today = datetime.datetime.now(datetime.timezone.utc) # retrieve todays exact date in datetime format #.date()

        closest_settle_to_today = min(historical_Data_options_maturity_dates_call_lasttradedates_datetime, key=lambda x: abs(x - static_Date_today)) #use lambda function, takes the abs value distance from each date in list and todays date and returns the smsallest one
        historical_Data_calls_with_settle_date_closest_to_today = historical_Data_options_maturity_dates.calls[historical_Data_options_maturity_dates.calls['lastTradeDate'] == closest_settle_to_today]
        print(historical_Data_calls_with_settle_date_closest_to_today)
        #strike price K
        option_Data_strike_price_K = historical_Data_calls_with_settle_date_closest_to_today.iloc[0,2] #first row and third column from left
        strike_Price_K_dict[f'{user_Input_list[i]}'] = option_Data_strike_price_K
        
        #last market price of 
        option_Data_last_market_price = historical_Data_calls_with_settle_date_closest_to_today.iloc[0,3]
        option_Market_price_dict[f'{user_Input_list[i]}'] = option_Data_last_market_price
        
    # engage in black sholes iteritive simulations to converge upon implied volitility
    implied_Vol_dict = {}
    implied_Vol_cummulative_numerator = 0
    for i in range(len(user_Input_list)):
        time_Factor_T = 0.5 #half a year or 6 months 
        tolerance = 0.00001 # this is the valuewe are converging on, realistically it should be 0 but for simulation purposes use a small positive number to avoid numerical errors or singularieties 
        vol_Old = 0.3      # also known as 30% --> this is our initital guess
        max_Iterations = 200 # reasonable nunber to avoid an infintie loop in case we dont converge upon a solution in case there isnt a defined zero in the fuction that you specified (i.e market price is unreasonable far away from black sholes price)
        option_Type = "c"
        ten_Year_rfrate = ten_Year_rfrate / 100 #adjust for bs input format

        for j in range(max_Iterations):

            #black_sholes_call_price w/ old volitility 
            d1_Oldvol = (np.log(last_Market_price_S0_dict[f'{user_Input_list[i]}'] / strike_Price_K_dict[f'{user_Input_list[i]}']) + (ten_Year_rfrate + (vol_Old**2) / 2) * time_Factor_T) / (vol_Old * (time_Factor_T**(1/2)))
            d2_Oldvol = d1_Oldvol - vol_Old * (time_Factor_T**(1/2))
            nd1_Oldvol = norm.cdf(d1_Oldvol)
            nd2_Oldvol = norm.cdf(d2_Oldvol)
            first_Term_Oldvol = last_Market_price_S0_dict[f'{user_Input_list[i]}'] * nd1_Oldvol #So*nd1 
            second_Term_Oldvol = strike_Price_K_dict[f'{user_Input_list[i]}']*(math.exp(-ten_Year_rfrate*time_Factor_T))*nd2_Oldvol
            bs_Price_Oldvol = first_Term_Oldvol - second_Term_Oldvol

            #vega or function prime w/ old volitility - finite differences method
            h = 1e-5 #step for finite differnces 
            nd1_plus_h = norm.cdf(d1_Oldvol + h) #cdf with h increment 
            nd1_no_h = norm.cdf(d1_Oldvol) # cdf without h incremnet
            nd1_Prime = (nd1_plus_h - nd1_no_h) / h #finite difference apporximation (the change in CDF due to a small change in vol)
            function_Prime = ((last_Market_price_S0_dict[f'{user_Input_list[i]}']*((time_Factor_T)**(1/2))) * nd1_Prime) * 100 #vega calc
            if abs(function_Prime) < tolerance: # to avoid a scalar issue
                break

            #function (BS call price with old vol - market price of option)
            function = bs_Price_Oldvol - option_Market_price_dict[f'{user_Input_list[i]}']

            #new vol is updated each iteration
            vol_New = vol_Old - function / (function_Prime) #+ 1e-8) # add a small value to avoid scalar issues in division

            #new black sholes w/ new volitility point Xn+1
            d1_Newvol = (np.log(last_Market_price_S0_dict[f'{user_Input_list[i]}'] / strike_Price_K_dict[f'{user_Input_list[i]}']) + (ten_Year_rfrate + (vol_New**2) / 2) * time_Factor_T) / (vol_New * (time_Factor_T**(1/2)))
            d2_Newvol = d1_Newvol - vol_New * (time_Factor_T**(1/2))
            nd1_Newvol = norm.cdf(d1_Newvol)
            nd2_Newvol = norm.cdf(d2_Newvol)
            first_Term_Newvol = last_Market_price_S0_dict[f'{user_Input_list[i]}'] * nd1_Newvol #So*nd1 
            second_Term_Newvol = strike_Price_K_dict[f'{user_Input_list[i]}']*(math.exp(-ten_Year_rfrate*time_Factor_T))*nd2_Newvol
            bs_Price_Newvol = first_Term_Newvol - second_Term_Newvol

            """
            iteritively solve through teh break condition, such that once tolerance condition is met, loop breaks and vol_New considered new volitility. Imporivng in the sense that the new_Vol becomes old 
            estiamteand we guess again (iteritvel refining estimate based in info avaialbe above). It refines the answer based on teh old. The hope is that each guess brings it closer
            """
        
            if (abs(vol_Old - vol_New) < tolerance) or (abs(bs_Price_Newvol-option_Market_price_dict[f'{user_Input_list[i]}']) < tolerance):
                break

            vol_Old = vol_New

        #individual to be appedned to dictionary
        implied_Vol = vol_New
        implied_Vol_dict[f'{user_Input_list[i]}'] = round(implied_Vol * 100,3)                     
    
    return implied_Vol_dict



#capm expected returns
def capm_Expected_returns_annual_individual(ten_Year_rfrate,beta_Dict,mean_Dict_annual, user_Input_list):
    
    #dictionary establish
    capm_Dict_individual_dict = {}

    for i in range(len(user_Input_list)):
        capm_Val_individual = beta_Dict[f'{user_Input_list[i]}'] * (mean_Dict_annual['^GSPC'] - ten_Year_rfrate) + ten_Year_rfrate
        capm_Dict_individual_dict[f'{user_Input_list[i]}'] = round(capm_Val_individual,3)

    return capm_Dict_individual_dict



#forward sharpe ratio 
def forward_Sharpe_ratio(ten_Year_rfrate, capm_Dict_individual_dict, implied_Vol_dict, user_Input_list):

    #establish dictionary
    forward_Sharpe_ratio_dict = {}

    for i in range(len(user_Input_list)):
        forward_Sharpe_val = (capm_Dict_individual_dict[f'{user_Input_list[i]}'] - ten_Year_rfrate) / implied_Vol_dict[f'{user_Input_list[i]}']
        forward_Sharpe_ratio_dict[f'{user_Input_list[i]}'] = round(forward_Sharpe_val,3)
    print(forward_Sharpe_ratio_dict)
    return forward_Sharpe_ratio_dict



#lstm portfolio model
def lstm_Predictive_timeseries(historical_Dataframe,user_Input_list):
    
    # calc cummulative performance - skip the market proxy gspc 
    historical_Dataframe["Cummulative Performance"] = historical_Dataframe.loc[:, historical_Dataframe.columns != '^GSPC'].sum(axis=1) / (len(user_Input_list))
    #create index and set cummulative returns to numeric
    historical_Dataframe.index = pd.to_datetime(historical_Dataframe.index.date) 
    historical_Dataframe['Cummulative Performance'] = pd.to_numeric(historical_Dataframe['Cummulative Performance'], errors='coerce')
    #dropna and linear interprolate other missing values
    historical_Dataframe.dropna()
    historical_Dataframe = historical_Dataframe.interpolate()
    print(historical_Dataframe)
    
    """
    Below we create our x and y variables. x_Input_list or our batch item inputs are the pre-existing datapoints we are using to predict a y outcome. We will later reference the outcome varaibles for y to determine the accuracy of our model 
    - the number of inputs or item into our batch will be 5. As the sequential dependanciees of financiaol data are often very short provided the physological aspect of market pricing and reactionary tendancies of the general public, 5 data points should be sufficient 
    - see below we are predicting t+2 with t+5 input. this is unconvential and does not follow a symetric pattern. Typically one would use t+5 input to predict the next data point t+6 and so on - but can be valid under certain cercumstances if leading to a more accurate model as:
        - note the underlying concern here is that we could experience data leakage as when we are training the model we are using data that would not technically be available yet (t+5) which could overinflate teh accuracy of the model. However it is importnant to note that we are merely training the model
        and allowing it to learn historical patterns (even t+5 going past t+2 in historical context would still be available data in a real wiorld scenario). Wghat is important is how the model performs on the testing data having fitted and trained teh model based of the strucutre and method the model was trained off of 
        with training data. Looking to test results we see the model is still quite accurate despite the unconventional training method as we see that with test data with numerous equity portfolio combinations (different holdings) an average value loss (MSE) after 20 epochs (choosing the best model) of 1.0565. As a measure of loss between predicitons and actual testing data
        in which with our data we see a relatively low target variable (teh range between data values are relatively small as we are dealing with daily returns), an average MSE on teh traing and validation data is relatively low/ acceptable
        - In real word scenarios the model may benfit from unconvential input output patterns
        - As we are leveraging future context (t+5) beyond t+2, we are providing the model with future context that has occured past t+2 (our predicition point) in our training data. This can allow
        for the model to capture valuble trends and patterns past t+2, allowing for a better predciiton
    """
    #create input and output
    number_Of_inputs = 5 # batch or input size of 5 due to short-term sequential dependencies in financial markets
    number_Of_feautures = 1 # 1 feature as we are focusing only on cummulative returns 
    historical_Dataframe_cumm = historical_Dataframe[['Cummulative Performance']] # isolate and create new dataframe... double bracket to ensure dataframe fromat and ability to trandofrm to 2D... avoid converting to series
    historical_Dataframe_cummulative_perf_to_np = historical_Dataframe_cumm.to_numpy() # transform to a numpy array |transforms to a 2D array additonally 
    
    x_Input_list = []
    y_State_list = []

    for i in range(len(historical_Dataframe_cummulative_perf_to_np)-number_Of_inputs): # maintain number of inputs for sequential dependancy for purpose of ensuring we do not go out of bounds
        row_Batch = [[a] for a in historical_Dataframe_cummulative_perf_to_np[i:i+number_Of_inputs]] 
        x_Input_list.append(row_Batch)

        label = historical_Dataframe_cummulative_perf_to_np[i+2]
        y_State_list.append(label)

    #convert back to array to develop and allow for formation of shape
    x_Input_list_array = np.array(x_Input_list)
    y_Input_list_array = np.array(y_State_list)

    #training and test
    list_Point_X = len(x_Input_list_array) // 3
    list_Point_Y = len(y_Input_list_array) // 3
   
    train_Data_x = np.array(x_Input_list[:list_Point_X]) # train is first 1/3
    train_Data_y = np.array(y_State_list[:list_Point_Y])

    validation_Data_X = np.array(x_Input_list[list_Point_X:2*list_Point_X]) #validation is middle 1/3
    validation_Data_y = np.array(y_State_list[list_Point_Y:2*list_Point_Y])

    test_Data_x = np.array(x_Input_list[list_Point_X:]) # test data is last 1/3, overlap on validation
    test_Data_y = np.array(y_State_list[list_Point_Y:])
 
    #define model 
    """
    number_Of_inputs = 10 #number of previous data points we use as inputs to produce our state - 
    number_Of_feautures = 1 # we are dealing with a univariant time series (cummulative returns) so feautures should be set to 1
    """
    lstm_Model = Sequential()
    lstm_Model.add(InputLayer((number_Of_inputs, number_Of_feautures))) # define the shape
    lstm_Model.add(LSTM(100)) # 100 nuerons | more neurons for more complexity | modern complexity, in context of problem 100 should be sufficient
    lstm_Model.add(Dense(8, 'relu')) # reltive linear unit activiation, good for handling complex data patterns
    lstm_Model.add(Dense(1, 'linear')) # output layer | linear makes good for regression tasks where model expected to predict continuous outpuit
    print(lstm_Model.summary())

    #callback 
    cp = ModelCheckpoint('lstm_Model/', save_best_only=True) # save only the model with the lowest validation loss or error

    # #compile - save only best model to lstm_model directory
    lstm_Model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()]) # keep learning rate (high number faster model decreases loss) small to find maximums accurately
    
    #pass train data 

    """
    When looking to how amny times we should run the model through teh training data (epochs), we see that from graphicing the loss that loss substancially drops in exponential 
    fashion, to avoid an innacurate model, overtraining and maintian computational efficiency e shall keep the model at 20 epochs

    #     loss_per_epoch = lstm_Model.history.history['loss']
    #     plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
    #     plt.show()

    below we see that when we fit the model, we use training data x and y to observe previous data(x) and previous outcomes(y) to learn how historical datas has interacted with itself. 
    The validation data is then used to validate the model duyring epochs, validate accuracy through MSE, and act as a second check and balance to the traingin data. 
    """

    #run model through training data 20 times (20 epochs)
    epoch_run_through = lstm_Model.fit(train_Data_x, train_Data_y,validation_data=(validation_Data_X,validation_Data_y), epochs=20, callbacks=[cp])
    train_loss = round(epoch_run_through.history['loss'][-1],3)
    train_root_mse = round(epoch_run_through.history['root_mean_squared_error'][-1],3)
    
    #load/ model with teh lowest validation loss
    lstm_Model = load_model('lstm_Model/')

    train_Predictions = lstm_Model.predict(test_Data_x).flatten() # remove inner bracakets 
    print(train_Predictions)

    train_Data_y_adjust_1d = test_Data_y.flatten() # turn y data to 1 dimension to place in dictionary dataframe

    historical_backtest_on_test_data = pd.DataFrame(data = {'Train Pred': train_Predictions, 'Actuals': train_Data_y_adjust_1d})
    print(historical_backtest_on_test_data)
    

    
    """
    Predict model forward

    grab the last batch or 5 datapoitns from test data and then predict the next value via "next_pred_role". Append this value to the numpy array and maintain the dimensionality or shape of the data. We can then remove the first element from the numpy array. 
    We can then predict the next value with an updated numpy array in which element -1 is the previous predicted value. This above effect will allow to take a rolling 5 day predicition window (keeping in mind an input or sequential dependancy of 5). 
    The predicted value(s) below are then appended to future_predictions.
    """
    #PREDICT FUTURE POINTS #3F dimensions 5,5,1 as it is univerariate cummulative returns. 
    num_future_predictions = 30
    future_predictions = []
    last_five_data_points = test_Data_x[-1:]

    #rolling prediction 
    for i in range(num_future_predictions):
        next_pred_role = lstm_Model.predict(last_five_data_points).flatten()
        future_predictions.append(next_pred_role)
        next_pred_role = np.append(last_five_data_points,next_pred_role)
        next_pred_role = next_pred_role[1:]
        next_pred_role = next_pred_role.reshape(1,5,1,1)
        #replace last_five_data_points with next_pred_role to enable batch role
        last_five_data_points = next_pred_role

    future_predictions = [x.item() for x in future_predictions]
    
    #GRAPH mext 30 days 
    next_30_days = [(dt.year,dt.month,dt.day) for dt in [datetime.datetime.today() + datetime.timedelta(days=i) for i in range(30)]]
    plt.plot(future_predictions)
    #  

    return next_30_days, future_predictions, historical_backtest_on_test_data, train_Predictions,train_Data_y_adjust_1d, train_loss,train_root_mse



#sma and ema trading strategy
def sma_ema_strat(next_30_days, future_predictions):
    
    sma_list = []
    ema_list = []
    strat_position = []

    ma_increment = 5

    #ema criteria
    ema_vals_list_first = sum(future_predictions[0:5])/ma_increment
    ema_list.append(ema_vals_list_first)

    for i in range(len(future_predictions)):
        sma_val = sum(future_predictions[i:i+5])/ma_increment
        sma_list.append(sma_val)

        ema_val = (future_predictions[i] - ema_list[i]) * (2 / (5+1)) + ema_list[i]
        ema_list.append(ema_val)

    #trading strat
    #append first  
    for i in range(len(future_predictions)):
        if future_predictions[i] < sma_list[i]:
            strat_position.append('SELL')
        elif future_predictions[i] > sma_list[i]:
            strat_position.append('BUY')
    #hold logic
    previous = None
    for i in range(len(strat_position)):
        if strat_position[i] == previous:
            strat_position[i] = 'HOLD'
        else:
            previous = strat_position[i] # None is assgined value of previous transaction as its at end 

    #graph results
    # plt.plot(future_predictions, label='Future Predict Actuals')
    # plt.plot(sma_list, label='5 Day SMA')
    # plt.plot(ema_list, label='5 Day EMA')
    # plt.legend()
    # plt.show(block=False)

    #dictionary comprehension 
    strat_position_dict = {round(future_predictions[i],3): strat_position[i] for i in range(len(strat_position))}
    print(strat_position_dict)

    return sma_list, ema_list, strat_position_dict



#main wrapper
def main():

    #run openGUI Input class
    run_GUI = openGUI() # define openGUI class as a variable
    run_GUI.root.mainloop() # start GUI event
    #retrieve input list from class above
    user_Input_list = run_GUI.user_Input_collection_list # append S&P500 to list for market proxy
    user_Input_list.append('^GSPC')
    user_Input_list_two = run_GUI.user_Input_collection_list_two
    print(user_Input_list)
    print(user_Input_list_two)

    #wrapping component
    historical_Dataframe, user_Input_list, closest_To_today, closest_To_date_fiveyearsago = historical_data_retrieve(user_Input_list, user_Input_list_two)
    ten_Year_rfrate = rf_Rate_scrape()
    historical_Dataframe, mean_Dict, stddev_Dict, sharpe_Dict, mean_Dict_annual,stddev_Dict_annual = historical_Stats(historical_Dataframe, user_Input_list, ten_Year_rfrate)
    min_Val_cummulative, max_Val_cummulative, mean_Val_cummulative, mean_Val_cummulative_annualized, stddev_Val_cummulative, stddev_Val_cummulative_annualized, sharpe_Ratio_cummulative_annualized = historical_Stats_cummulative(historical_Dataframe, user_Input_list, ten_Year_rfrate)
    beta_Dict, weighted_beta = beta_calc(historical_Dataframe, user_Input_list, mean_Dict,stddev_Dict)
    historical_Dataframe_correlation_matrix = correlation_Matrix(historical_Dataframe, user_Input_list)
    implied_Vol_dict = black_Sholes_newton_impliedvol(user_Input_list, ten_Year_rfrate, closest_To_today, closest_To_date_fiveyearsago)
    capm_Dict_individual_dict = capm_Expected_returns_annual_individual(ten_Year_rfrate,beta_Dict,mean_Dict_annual, user_Input_list)
    forward_Sharpe_ratio_dict = forward_Sharpe_ratio(ten_Year_rfrate, capm_Dict_individual_dict, implied_Vol_dict, user_Input_list)
    next_30_days, future_predictions, historical_backtest_on_test_data,train_Predictions,train_Data_y_adjust_1d,train_loss,train_root_mse = lstm_Predictive_timeseries(historical_Dataframe,user_Input_list)
    sma_list,ema_list,strat_position_dict = sma_ema_strat(next_30_days, future_predictions)
    
    #GUI output
    root = tk.Tk()
    app = display_GUI(root)
    app.create_historical_page(ten_Year_rfrate,mean_Dict,stddev_Dict,sharpe_Dict,mean_Dict_annual,stddev_Dict_annual,min_Val_cummulative,max_Val_cummulative,mean_Val_cummulative,
                               stddev_Val_cummulative,stddev_Val_cummulative_annualized,sharpe_Ratio_cummulative_annualized,beta_Dict,weighted_beta,historical_Dataframe_correlation_matrix,mean_Val_cummulative_annualized)
    
    app.create_projection_page(implied_Vol_dict,capm_Dict_individual_dict,forward_Sharpe_ratio_dict,strat_position_dict,historical_backtest_on_test_data,future_predictions,sma_list,ema_list,train_Predictions,train_Data_y_adjust_1d,train_loss,train_root_mse)
    
    root.mainloop()



#call main
main()