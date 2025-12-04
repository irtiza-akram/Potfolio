# Install required Python libraries for data preprocessing, model training, and optional web deployment

!pip install nltk scikit-learn pandas numpy flask-ngrok

# prompt: instal sklearn for feature extration text

!pip install -U scikit-learn

# Import necessary libraries for data handling, visualization, and ML

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

# Load the datasets (True and Fake news)

true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')


# Preview the first few rows of the real news dataset

true.head()



# Preview the first few rows of the fake news dataset

fake.head()

# Add a 'label' column: 1 for True, 0 for Fake

fake['label']= 0
true['label']= 1

# checking the rows and colums in data sets
fake.shape, true.shape

# Create manual testing dataset and remove these samples from the main datasets.

fake_manual_testing = fake.tail(10) # Extract the last 10 rows of the 'fake' DataFrame for manual testing
for i in range(23480,23470,-1):    # Iterate in reverse order to remove the last 10 rows from 'fake'
    fake.drop([i], axis=0, inplace=True)  # Remove rows by index to avoid issues with shifting indices

true_manual_testing = true.tail(10) # Extract the last 10 rows of the 'true' DataFrame for manual testing
for i in range(21416,21406,-1):  # Iterate in reverse order to remove the last 10 rows from 'true'
    true.drop([i], axis=0, inplace=True)   # Remove rows by index to avoid issues with shifting indices

# Add a 'label' column to the manual testing datasets: 0 for Fake, 1 for True


fake_manual_testing['label']= 0   # Assign label 0 to all rows in 'fake_manual_testing'
true_manual_testing['label']= 1   # Assign label 1 to all rows in 'true_manual_testing'



# Display the first 5 rows of the 'fake_manual_testing' DataFrame.
# This allows for a quick preview of the data that was set aside
# for manual testing of the fake news detection model.

fake_manual_testing.head()




# Display the first 5 rows of the 'true_manual_testing' DataFrame.
# This allows for a quick preview of the data that was set aside
# for manual testing of the fake news detection model.

true_manual_testing.head()


# Concatenate the fake and true manual testing DataFrames into a single DataFrame
manual_testing = pd.concat([fake_manual_testing, true_manual_testing], axis=0)

# Save the combined manual testing data to a CSV file named 'manual_testing.csv'
manual_testing.to_csv('manual_testing.csv')


# Assuming 'fake' and 'true' are DataFrames containing fake and real news data respectively

# Concatenate the 'fake' and 'true' DataFrames vertically
# to create a single DataFrame called 'df_marge'
# axis=0 specifies that the DataFrames should be stacked on top of each other
df_marge = pd.concat([fake, true], axis=0)


# Display the first 5 rows of the 'df_marge' DataFrame
# to preview the combined dataset
df_marge.head()



# Display the last 5 rows of the 'df_marge' DataFrame
# to preview the combined dataset
df_marge.tail()


# Display the column names of the 'df_marge' dataframe.
df_marge.columns


# Drop the 'title' , 'date' and 'subject' columns from both datasets
df = df_marge.drop(['title', 'subject', 'date'], axis = 1)

# Check for missing values (nulls) in the DataFrame 'df'.
# This is crucial for data cleaning as missing values can
# negatively impact model training and accuracy.
# The .isnull() method identifies missing values,
# and .sum() calculates the total for each column.
df.isnull().sum()



# Shuffle the DataFrame 'df' in-place to randomize the order of rows.
# This is important for avoiding potential biases during model training
# that could arise from the original order of the data.
# 'frac=1' indicates sampling 100% of the data, effectively shuffling all rows.
df = df.sample(frac = 1)


# Display the first 5 rows of the DataFrame for quick preview
df.head()




# Reset the index
df.reset_index(inplace = True)
# Remove the old index
df.drop(['index'], axis = 1, inplace = True)


# for the quick preview
df.head()


# Cleans and preprocesses text for NLP tasks by:
    # - Converting to lowercase
    # - Removing brackets, URLs, HTML tags, and punctuation
    # - Replacing non-alphanumeric characters with spaces
    # - Removing words containing numbers

def wordopt(text):
      text = text.lower()
      text = re.sub('\[.*?\]', '', text)
      text = re.sub("\\W", " ", text)
      text = re.sub('https?://\S+|www\.\S+', '', text)
      text = re.sub('<.*?>+', '', text)
      text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
      text = re.sub('\n', '', text)
      text = re.sub('\w*\d\w*', '', text)
  return text


# storing clean dataframe df

df["text"] = df["text"].apply(wordopt)

# Define feature variable (X) and target variable (y) for the model.
# X will contain the text content of the news articles.
# y will contain the labels (0 for fake, 1 for real).

x = df["text"]
y = df['label']

# Split the data into training and testing sets
# x: Input features (news text content)
# y: Target variable (news labels - 0 for fake, 1 for real)
# test_size: Proportion of data to include in the test set (25% in this case)
# x_train, y_train: Training data (features and labels)
# x_test, y_test: Testing data (features and labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Import the TfidfVectorizer class from scikit-learn for text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer object with default parameters
vectorization = TfidfVectorizer()

# Fit the vectorizer to the training data (x_train) and transform it into a numerical representation
# This learns the vocabulary and IDF weights from the training data and creates the document-term matrix
xv_train = vectorization.fit_transform(x_train)

# Transform the testing data (x_test) using the fitted vectorizer
# This ensures that the testing data is represented using the same vocabulary and IDF weights as the training data
xv_test = vectorization.transform(x_test)


# Import the Logistic Regression model from scikit-learn
from sklearn.linear_model import LogisticRegression

# Create an instance of the Logistic Regression model
# We're using the default parameters for now, but these can be customized
LR = LogisticRegression()

# Train the model using the training data
# xv_train: The features (text content transformed into numerical representation) of the training data
# y_train: The corresponding labels (0 for fake, 1 for real) for the training data
# The 'fit' method adjusts the model's internal parameters to learn from the training data
LR.fit(xv_train, y_train)

# Import the Logistic Regression model from scikit-learn
from sklearn.linear_model import LogisticRegression

# Create an instance of the Logistic Regression model
# We're using the default parameters for now, but these can be customized
LR = LogisticRegression()

# Train the model using the training data
# xv_train: The features (text content transformed into numerical representation) of the training data
# y_train: The corresponding labels (0 for fake, 1 for real) for the training data
# The 'fit' method adjusts the model's internal parameters to learn from the training data
LR.fit(xv_train, y_train)

# Import the Logistic Regression model from scikit-learn
from sklearn.linear_model import LogisticRegression

# Create an instance of the Logistic Regression model
# We're using the default parameters for now, but these can be customized
LR = LogisticRegression()

# Train the model using the training data
# xv_train: The features (text content transformed into numerical representation) of the training data
# y_train: The corresponding labels (0 for fake, 1 for real) for the training data
# The 'fit' method adjusts the model's internal parameters to learn from the training data
LR.fit(xv_train, y_train)


# Evaluate the performance of the Logistic Regression model on the test set
# by printing a classification report.
# This report provides key metrics such as precision, recall, F1-score,
# and support for each class (Fake and Real news).
# It helps assess the model's ability to correctly classify news articles.

print(classification_report(y_test, pred_lr))


# Import, create, and train a Bernoulli Naive Bayes model
from sklearn.naive_bayes import BernoulliNB

NB = BernoulliNB() # Create a Bernoulli Naive Bayes object
NB.fit(xv_train, y_train) # Train the model using training data


# Use the trained Naive Bayes model (NB) to predict labels for the test set (xv_test)
# and store the predictions in the 'pred_nb' variable.
pred_nb = NB.predict(xv_test)

# Calculate and print (implicitly) the accuracy score of the Naive Bayes model on the training data (xv_train, y_train).
# This provides an evaluation of how well the model learned from the training data.
NB.score(xv_train, y_train)


# Evaluate the performance of the Naive Bayes model on the test set
# by printing a classification report.
# This report provides key metrics such as precision, recall, F1-score,
# and support for each class (Fake and Real news).
# It helps assess the model's ability to correctly classify news articles.
print(classification_report(y_test, pred_nb))


# Import the DecisionTreeClassifier class from scikit-learn's tree module
from sklearn.tree import DecisionTreeClassifier

# Create a DecisionTreeClassifier object with default settings
# This will be our decision tree model
DT = DecisionTreeClassifier()

# Train the decision tree model using the training data
# xv_train: The features (preprocessed text data)
# y_train: The corresponding labels (0 for fake, 1 for real)
DT.fit(xv_train, y_train)


# Make predictions on the test set using the trained Decision Tree model
# xv_test: The features (preprocessed text data) of the test set
# pred_dt: A variable to store the predicted labels (0 for fake, 1 for real)
pred_dt = DT.predict(xv_test)

# Evaluate the performance of the Decision Tree model on the training data
# xv_train: The features (preprocessed text data) of the training set
# y_train: The true labels for the training set
# This line calculates and prints (implicitly) the accuracy score
DT.score(xv_train, y_train)


# Evaluate the performance of the Decision Tree model on the test set
# by printing a classification report.
# This report provides key metrics such as precision, recall, F1-score,
# and support for each class (Fake and Real news).
# It helps assess the model's ability to correctly classify news articles.

print(classification_report(y_test, pred_dt))


# Import the RandomForestClassifier class from scikit-learn's ensemble module
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier object with default settings
# This will be our Random Forest model
RFC = RandomForestClassifier()

# Train the Random Forest model using the training data
# xv_train: The features (preprocessed text data)
# y_train: The corresponding labels (0 for fake, 1 for real)
RFC.fit(xv_train, y_train)


# Make predictions on the test set using the trained Random Forest model
pred_rfc = RFC.predict(xv_test)

# Evaluate the model's performance on the training data using the score method
# This gives you an idea of how well the model learned from the training data
# but doesn't necessarily reflect its performance on unseen data (test set)
RFC.score(xv_train, y_train)


# Evaluate the performance of the Random Forest model on the test set
# by printing a classification report.
# This report provides key metrics such as precision, recall, F1-score,
# and support for each class (Fake and Real news).
# It helps assess the model's ability to correctly classify news articles.

print(classification_report(y_test, pred_rfc))   # Print the classification report

# Define a function to convert numerical labels (0 or 1) to text labels ("Fake News" or "Real News")
def output_lable(n):
    if n == 0:
        return "Fake News" # If n is 0, it's fake news
    elif n == 1:
        return "Real News"  # If n is 1, it's real news

# Define a function for manual testing of news articles
def manual_testing(news):

     # Create a dictionary to store the news article text
    testing_news = {"text":[news]}
     # Convert the dictionary to a Pandas DataFrame
    new_def_test = pd.DataFrame(testing_news)
     # Apply the wordopt function to clean and preprocess the text
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
     # Extract the preprocessed text
    new_x_test = new_def_test["text"]
    # Vectorize the text using the previously fitted vectorizer
    new_xv_test = vectorization.transform(new_x_test)
    # Make predictions using the four trained models
    pred_LR = LR.predict(new_xv_test)  # Logistic Regression prediction
    pred_NB = NB.predict(new_xv_test)  # Naive Bayes prediction
    pred_DT = DT.predict(new_xv_test)  # Decision Tree prediction
    pred_RFC = RFC.predict(new_xv_test) # Random Forest prediction
    # Print the predictions of all four models with readable labels
    return print("\n\nLR Prediction: {} \nNB Prediction: {} \nDT Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),  # Convert LR prediction to text label
                                                                                                              output_lable(pred_NB[0]), # Convert NB prediction to text label
                                                                                                              output_lable(pred_DT[0]), # Convert DT prediction to text label
                                                                                                              output_lable(n=pred_RFC[0])))  # Convert RFC prediction to text label


# Get news input from the user
news = str(input())

# Call the manual_testing function to classify the news
manual_testing(news)






