import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns  

# NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

# Streamlit app
st.title("Tweet Classification: Hate Speech Detection")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with 'tweet' and 'class' columns", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    # Map class labels
    data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive language", 2: "No Hate and Offensive Speech"})
    data = data[["tweet", "labels"]]

    # Display class distribution
    st.write("### Class Distribution")
    class_counts = data["labels"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="viridis")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class Labels")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Clean the tweets
    data["tweet"] = data["tweet"].apply(clean_text)

    # Prepare data for training
    x = np.array(data["tweet"])
    y = np.array(data["labels"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Train the classifier
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, clf.predict(x_train)) * 100
    test_accuracy = accuracy_score(y_test, clf.predict(x_test)) * 100

    st.write("### Model Performance")
    st.write(f"Accuracy on training data: **{train_accuracy:.2f}%**")
    st.write(f"Accuracy on test data: **{test_accuracy:.2f}%**")

    # Predict user input
    user_input = st.text_area("Enter a tweet for classification:")
    if st.button("Predict"):
        if user_input:
            cleaned_input = clean_text(user_input)
            vectorized_input = cv.transform([cleaned_input])
            prediction = clf.predict(vectorized_input)
            st.write(f"### Prediction: **{prediction[0]}**")
        else:
            st.write("Please enter a tweet to classify.")
