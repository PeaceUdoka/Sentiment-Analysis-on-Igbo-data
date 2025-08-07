# Sentiment Analysis on Igbo Data

### Objective: Build a model that can classify sentiments on news articles as positive, neutral, or negative.

### Data: The data was obtained from NaijaSenti's manually annotated Igbo tweets, available [here.](https://github.com/hausanlp/NaijaSenti/tree/main)

### Requirements: To replicate and build upon this work, do the following

| Step | Command Example |
|-----------------------------|---------------------------------------------------------| 
| Clone repo | `git clone https://github.com/PeaceUdoka/Igbo-AI-translator.git` | 
| Create virtual env | `python -m venv venv` | 
| Activate virtual env (Win) | `venv\Scripts\activate` | 
| Activate virtual env (Mac/Linux) | `source venv/bin/activate` | 
| Install dependencies | `pip install -r requirements.txt` | 
| To run Streamlit app | `streamlit run app.py` | 
| Deactivate virtual env | `deactivate`

### Data Wrangling and Preprocessing

The initial phase involved preparing the raw data for model training. This began with loading and combining the training and development datasets. An initial inspection was performed to understand the data's structure, sentiment distribution, and identify any missing values.
Text cleaning was a crucial step, involving the removal of punctuation, special characters, links, and HTML tags, along with converting text to lowercase. To enhance the focus on meaningful words, a list of Igbo-specific stop words was created and refined, combined with common English stop words, and then removed from the text. The impact of emojis on sentiment analysis was investigated by creating four different versions of the text data: retaining emojis as is, removing them, replacing them with text descriptions, and moving them to the end of the text with descriptions. Finally, the categorical sentiment labels were converted into a numerical format suitable for model input.

### Vectorization

To enable machine learning and deep learning models to process the text data, it was converted into numerical representations using TF-IDF vectorization. This technique captures the importance of words within documents and across the entire dataset, considering both single words (unigrams) and pairs of adjacent words (bigrams) and limiting the features to the most relevant 500. This process was applied to all four versions of the text data resulting from the emoji handling exploration.

### Machine Learning Model Building and Evaluation

Several traditional machine learning models were trained and evaluated to determine their effectiveness for Igbo sentiment analysis. Logistic Regression, Random Forest Classifier, Gaussian Naive Bayes, and Decision Tree Classifier were applied to each of the four datasets with different emoji handling strategies. Each model's performance was assessed using standard classification metrics and confusion matrices to understand their strengths and weaknesses, particularly in handling the minority negative sentiment class. It was observed that the Logistic Regression model, trained on the data where emojis were kept as they were, achieved the highest accuracy (75%) among these models. The experiment on emoji handling strategies indicated that altering emojis did not lead to significant improvements in the performance of these machine learning models. In some, it even performed worse.

### Deep Learning (DL) Model Building and Evaluation

A deep learning approach was also explored by building and training a simple Sequential model consisting of Dense layers and Dropout. The numerical labels were one-hot encoded, and the data was split into training, validation, and test sets for rigorous evaluation. The model was compiled with an Adam optimizer and categorical crossentropy loss. Training incorporated callbacks for saving the best model, reducing the learning rate, and early stopping to prevent overfitting. The deep learning model was trained and evaluated on the four datasets with varying emoji treatments. The best performance for this deep learning model (75.85%) was achieved when using the data with emojis left as they were, suggesting that for this architecture, preserving emojis was the most effective strategy.

### Fine-Tune the AfriBERTa Model

Leveraging the power of pre-trained language models, the [Davlan AfriBERTa Large model](https://huggingface.co/castorini/afriberta_large) was fine-tuned for the Igbo sentiment analysis task. The cleaned data was converted into a Hugging Face Dataset object and split into training, validation, and test sets. The pre-trained tokenizer and model were loaded, and all layers except the classification head were frozen to focus training on the task-specific layer. The data was tokenized, and a data collator was used for efficient batching and dynamic padding. The model was trained using the Hugging Face Trainer with specified training arguments and an early stopping callback. The fine-tuned model was evaluated on the test set, yielding an accuracy of 63.6%

The LogReg, DL and and fine_tuned afriberta models were saved for further predictions.

### Predicting Scraped Data

From deduction, news articles should be neutral. The trained models were applied to a dataset of scraped Igbo news articles from BBC and IgboRadio to assess their sentiment. The scraped text data underwent the same cleaning and stop word removal process as the training data.
1.	Logistic Regression Prediction: The Logistic Regression model and TF-IDF vectorizer were loaded and used to predict the sentiment of the cleaned and vectorized scraped news articles. 97.7% of the news data was predicted as neutral.
2.	Deep Learning Model Prediction: The deep learning model was loaded and used to predict the sentiment of the vectorized scraped news articles. All articles were predicted as having a neutral sentiment.
3.	Refined BERT Prediction: The fine-tuned BERT model and tokenizer were loaded and used to predict the sentiment of the tokenized scraped news articles. All articles were predicted as negative. Further work and analysis is recommended on improving the model.
4.	Davlan AfriBERTa Large Model (Sentiment Analyisi Pipeline) Prediction: A pre-trained AfriBERTa model for sentiment analysis was loaded and used to predict the sentiment of the cleaned and stop-word removed scraped news articles. The model predicted all articles as neutral. 
