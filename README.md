# Can comments from Reddit and Twitter be used to predict next day stock prices?

## Introduction
This github projects provides an outline of my dissertation which I did as part of my MSc Data Science course. The motivation for this projects came from the events in early 2021 when retail investors on Reddit, specifically on r/wallstreetbets community, traded against positions of short-selling hedge funds and paved the way for the closure of a few hedge funds.

## Abstract
This study looked at how the investor sentiment from the largest investment community on Reddit compares to that of Twitter in stock price prediction. At a high level, the study also analyses if Reddit is a credible source of investor sentiment by combining sentiments from Reddit with the stock price related technical inputs and using them to predict the stock prices through a Deep Neural Network model. Models which give the best results in recent times were used for both sentiment classification and stock price prediction tasks namely, Bidirectional Encoder Representations from Transformers (BERT) for sentiment classification and a hybrid Convolutional Long Short-Term Memory Neural Network (CNN-LSTM) for stock price prediction. Data from Reddit and Twitter were extracted and processed using python libraries. Four different BERT models, each trained with a different dataset was used to predict the sentiments of five stocks from Reddit and Twitter, after which they were used along with the technical data for each stock to predict the next day closing price via the CNN-LSTM network. Apart from using Root Mean Squared Error (RMSE) to compute the prediction accuracy, directional accuracy was also studied by computing the profits under two assumed scenarios, where the scenarios used the stock price predicted using sentiment data from Reddit and Twitter. The study found that sentiments from Reddit fared better in more cases when compared to that from Twitter. It was also found that, in some cases, combining data from Reddit and Twitter had better performance compared to the respective datasets in isolation.

## STEPS

In summary the steps involve,
 - Extracting required data from Reddit and Twitter
 - Preparing the extracted data so it can be used with BERT transformer
 - Use different variations of BERT transformer to extract daily sentiments from Reddit and Twitter
 - Extract and prepare technical data for selected stocks (stocks selected for price predictions)
 - Build a CNN-LSTM model and tune it to find the best hyperparameters
 - Use the CNN-LSTM model to predict the next day stock prices by using (technical stock data + sentiments) from previous day


### Step 1: Data Extraction

[![name](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/Images/img1.png)](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F1A_Extracting_data_from_reddit.ipynb)


1. Data from Reddit was collected from the community r/wallstreetbets using PMAW library in python. PMAW extracts Reddit data from pushshift prohject which archives Reddit data close to real-time. Though Reddit's own API can be used to extract data directly from Reddit, the rate limit imposed by Reddit makes it impossbile to extract a large amount of data in a time efficient manner. The python code for Reddit data extraction can be seen in,
_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F1A_Extracting_data_from_reddit.ipynb_

2. Before collecting data from Twitter, Reddit data was analyzed to select the few stocks for which the project would be carried out for. In other words, five stocks were selected from Reddit data for which proce predicton would be done. The five stocks were AMC, TSLA, AMD, BABA and DKNG.

3. Data from Twitter was extracted for the five selected stocks. The python library snscrape was used to extract data from Twitter. The python code for the same can be viewed at,
_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F1B_Extracting_data_from_twitter.ipynb_


### Step 2: Data Preparation


[![name](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/Images/img2.png)](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F2A_Prepare_reddit_data_for_BERT.ipynb)


In order to extract sentiments from Reddit and Twitter comments for each of the five stocks on a daily basis, BERT model was used. The data preparation steps prior to this such as removing special characters, double spaces, etc., and also trimming comments to 512 characters were done and can be seen in the below python codes,

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F2A_Prepare_reddit_data_for_BERT.ipynb_

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F2B_Prepare_twitter_data_for_BERT.ipynb_

### Step 3: Preparation of BERT model and sentiment classification of comments


[![name](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/Images/img3.png)](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F3A_Bert_sentiment_analysis_TwiiterAirline_Even.ipynb)



The BASE BERT UNCASED model from huggingface python library was used for this. BERT has to be finetuned with a local dataset prior to using it. Though that is the ideal procedure, the lack of labelled training data for Reddit meant that fine-tuning could only be done using open source twitter datasets. To overcome any impacts due to this, a total of four BERT models were used,
- Three of which were fine-tuned with datasets mentioned in the below Table
- The fourth model was chosen as the readily available finBERT model from huggingface library. This finBERT model was pre-trained and fine-tuned with financial corpus and financial sentiment classification.



References mentioned in the table:

_Kaggle, 2019. Twitter US Airline Sentiment. [Online]. Available at: https://www.kaggle.com/crowdflower/twitter-airline-sentiment?select=Tweets.csv [Accessed 23 November 2021]._
_Go, A., Bhayani, R. & Huang, L., n.d. Sentiment140. [Online]. Available at: http://help.sentiment140.com/for-students [Accessed 22 November 2021]._


After finetuning the BERT Models with respective datasets as shown in the above table, daily sentiments from both Reddit and Twitter were computed as the probability of positivity, neutrality and negativity. The respective python codes can be viewed in the below links,

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F3A_Bert_sentiment_analysis_TwiiterAirline_Even.ipynb

https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F3B_Bert_sentiment_analysis_TwiiterAirline_Uneven.ipynb

https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F3C_Bert_sentiment_analysis_Sentiment140.ipynb

https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F3D_finBERT_sentiment_analysis.ipynb_


### Step 4: Preparation of technical stock data


[![name](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/Images/img4.png)](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F4A_Stock_technical_data.ipynb)



In order to predict stock prices, along with the sentiments, a stock's technical data was also used. A total of 16 technical indicators were used. The indicators and their definitions are given in the below image. An important point to notice here is that the log return of previous day is also taken as one the technical inputs. Since stock prices are non-stationary it is a best practice to use the log returns instead of the raw previous day prices. The log return makes the stock price stationary. 

The different formulae for the indicators were taken from the following research papaer,
_Gao, T. & Chai, Y., 2018. Improving stock closing price prediction using recurrent neural network and technical indicators. Neural computation, 30(10), pp. 2833-2854._

The python code for this part can be seen in the below link,
_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F4A_Stock_technical_data.ipynb_


### Step 5: CNN-LSTM Model and tuning its hyperparameters 

From various recent studies it was seen that a combination of CNN-LSTM models performed best in stock price prediction problems. So a parameterized CNN-LSTM model was created using the Keras module in python. GridSearchCV module from scikit-learn python library was used to find the best hyperparameters for the CNN-LSTM model. This code can be viewed in,

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F5A_Hyperparameter%20tuning%20for%20each%20stock.ipynb_

_Note: This jupyter notebook is huge and hence may not open on git-hub interface. It is best to download this and view on your local machine._

### Step 6: Predictiing next day stock prices with CNN-LSTM

[![name](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/Images/img6.png)](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F6A_CNN_LSTM_Predictions_for_BERT1_data.ipynb)


Since we derived everyday sentiments from four BERT models, we have four sets of data [sentiment + technical indicator] for each of the five chosen stocks. The CNN-LSTM model was used with the best parameters found from step 5 and used to predict next day stock prices for each of the five stocks. These can be viewed in the following links,

https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F6A_CNN_LSTM_Predictions_for_BERT1_data.ipynb

https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F6B_CNN_LSTM_Predictions_for_BERT2_data.ipynb

https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F6C_CNN_LSTM_Predictions_for_BERT3_data.ipynb

https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F6D_CNN_LSTM_Predictions_for_BERT4_data.ipynb


### Step 7: Data visualizations

[![name](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/Images/img7.png)](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F7B_Dataset_Statistics.ipynb)

From each CNN-LSTM prediction, the root mean squared error was calculated to check how the prediction accuracy was. The different input data were also visualized to see if any patterns could be found with respect to the metric values.

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F7A_metrics_comparison.ipynb_

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F7B_Dataset_Statistics.ipynb_

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F7C_Dataset_Statistics_2.ipynb_

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F7D_Dataset_Counts_Per_Month.ipynb_



### Step 8: Performance estimation using practical scenarios


[![name](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/Images/img8.png)](https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F7E_Profit_Loss_Calculations.ipynb)


To have a practical comparison, two scenarios were formed and calculations were made to check to see which datasets would provide better profits if stock market investments were made using predictions from the different models. To contain the number of variables and models, the average profit by investing in all five stocks were calculated for data from each of the BERT Models.

The following section describes the two scenarios considered,
Scenario 1:
This scenario assumes a moderately experienced investor who does both buying and short selling of stocks. The investor would initially invest a sum of $100. Each day the investor decides using the predictions for the next day’s closing price. 
- If the predicted closing price for day n+1 is greater than day n, then the investor buys the stock at the current day’s closing price
- If the predicted closing price for day n+1 is less than day n, then the investor borrows the stock at the current day’s (day n) closing price and settles the borrowed stocks at the end of day n+1. This is nothing but a short sale. If the prediction is right, the investor earns the difference but if the prediction is wrong and the closing price increases the next day, the investor loses the difference.
- The profit and loss are cumulative across the entire timeline (i.e.) if the initial $100 increases to $120 by day 10, then the next move (buying or short selling) will involve $120.

Scenario 2:
This scenario assumes a novice investor who only buys the stock if the predicted price for day n+1 is higher than the closing price for day n. Like scenario 1, the profit and loss are cumulative across the entire timeline.
We calculated the average profit the investor would make under each scenario by investing $100 in each stock using sentiments from each BERT Model.


_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F7E_Profit_Loss_Calculations.ipynb_


### Step 9:

To understand the entire study in terms of a reasearch prespective and what research questions were answered by the study, it is highly recommended to read my entire dissertation,
_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/Full_Dissertation_Report.pdf_
