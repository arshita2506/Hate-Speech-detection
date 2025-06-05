# 🧐 Tweet Classification: Hate Speech Detection

This is a simple Streamlit web app that classifies tweets into three categories:

* **Hate Speech**
* **Offensive Language**
* **No Hate and Offensive Speech**

The model is trained using **Davidson et al.'s (2017) dataset** on hate speech detection and utilizes a basic NLP preprocessing pipeline with a **Decision Tree Classifier**.

---

## 🚀 Features

* Upload CSV files containing tweets and labels.
* Real-time classification of user-input tweets.
* Visualize class distribution.
* Display model accuracy on training and testing sets.

---

## 📁 Dataset

The app is built using the publicly available [Davidson Hate Speech Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language), which includes 24k+ tweets annotated into three classes:

* `0` → Hate Speech
* `1` → Offensive Language
* `2` → Neither (Clean)

Ensure your uploaded CSV file has at least the following columns:

* `tweet`
* `class` (numeric labels: 0, 1, 2)

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** for the web interface
* **Pandas / NumPy** for data handling
* **NLTK** for preprocessing (stopwords, stemming)
* **Scikit-learn** for vectorization, model training and evaluation
* **Seaborn / Matplotlib** for data visualization

---

## ⚙️ How It Works

1. **Text Cleaning**:
   Tweets are preprocessed by removing URLs, punctuation, digits, and stopwords, and then stemmed using `SnowballStemmer`.

2. **Vectorization**:
   Text data is converted into numerical features using `CountVectorizer`.

3. **Model**:
   A simple `DecisionTreeClassifier` is trained on the vectorized tweets.

4. **Prediction**:
   The trained model can classify new, unseen tweets in real-time.

---

## 🧪 Running the App Locally

1. **Clone this repository**

   ```bash
   git clone To https://github.com/arshita2506/Hate-Speech-detection.git
   cd hate-speech-detection
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required libraries**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

---

## 📌 To Do

* Add more sophisticated models (e.g., Logistic Regression, SVM)
* Integrate TF-IDF vectorizer
* Model persistence with pickle
* Add confusion matrix and classification report
* Improve UI with Streamlit widgets

---

## 👨‍🔬 Author

**Arshita Garg**

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
