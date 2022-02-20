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

1. Data from Reddit was collected from the community r/wallstreetbets using PMAW library in python. PMAW extracts Reddit data from pushshift prohject which archives Reddit data close to real-time. Though Reddit's own API can be used to extract data directly from Reddit, the rate limit imposed by Reddit makes it impossbile to extract a large amount of data in a time efficient manner. The python code for Reddit data extraction can be seen in,
_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F1A_Extracting_data_from_reddit.ipynb_

2. Before collecting data from Twitter, Reddit data was analyzed to select the few stocks for which the project would be carried out for. In other words, five stocks were selected from Reddit data for which proce predicton would be done. The five stocks were AMC, TSLA, AMD, BABA and DKNG.

3. Data from Twitter was extracted for the five selected stocks. The python library snscrape was used to extract data from Twitter. The python code for the same can be viewed at,
_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F1B_Extracting_data_from_twitter.ipynb_

### Step 2: Data Preparation
In order to extract sentiments from Reddit and Twitter comments for each of the five stocks on a daily basis, BERT model was used. The data preparation steps prior to this such as removing special characters, double spaces, etc., and also trimming comments to 512 characters were done and can be seen in the below python codes,

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F2A_Prepare_reddit_data_for_BERT.ipynb_

_https://github.com/Blindusername001/Can-Reddit-and-Twitter-data-be-used-to-predict-stock-price-movements-/blob/main/F2B_Prepare_twitter_data_for_BERT.ipynb_

### Step 3: Preparation of BERT model
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
In order to predict stock prices, along with the sentiments, a stock's technical data was also used. A total of 16 technical indicators were used. The indicators and their definitions are given in the below image. An important point to notice here is that the log return of previous day is also taken as one the technical inputs. Since stock prices are non-stationary it is a best practice to use the log returns instead of the raw previous day prices. The log return makes the stock price stationary. 

The different formulae for the indicators were taken from the following research papaer,
_Gao, T. & Chai, Y., 2018. Improving stock closing price prediction using recurrent neural network and technical indicators. Neural computation, 30(10), pp. 2833-2854._

