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
