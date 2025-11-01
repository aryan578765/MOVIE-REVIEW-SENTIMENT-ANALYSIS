---

# 🎬 IMDB Movie Review Sentiment Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-review-sentiment-analysis-77.streamlit.app/)

A complete end-to-end machine learning project that classifies IMDB movie reviews as either positive or negative. This project showcases the entire ML pipeline, from data preprocessing and model training to deploying an interactive web application using Streamlit.

## 🎯 Project Goal

The primary goal of this project was to build a robust sentiment analysis system capable of understanding the sentiment behind movie reviews. This involved handling noisy, real-world text data and building a model that generalizes well to unseen reviews.

## 🌐 Live Demo

Experience the live application here: **[https://movie-review-sentiment-analysis-77.streamlit.app/](https://movie-review-sentiment-analysis-77.streamlit.app/)**

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning Libraries:** Pandas, Scikit-learn, NLTK
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git & GitHub

## 📋 Project Workflow

This project follows a structured machine learning workflow:

1.  **Data Acquisition & Exploration:**
    -   Sourced the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle.
    -   Performed exploratory data analysis (EDA) to understand review length, class balance, and common words.

2.  **Text Preprocessing:**
    -   Cleaned raw text by removing HTML tags, punctuation, and converting to lowercase.
    -   Tokenized text and removed common English stopwords.
    -   Applied lemmatization to reduce words to their root form.

3.  **Feature Engineering:**
    -   Converted processed text into numerical features using **TF-IDF Vectorization**.

4.  **Model Training & Evaluation:**
    -   Trained multiple classification models, including **Logistic Regression** and **Naive Bayes**.
    -   Evaluated models using metrics like Accuracy, Precision, Recall, and F1-Score.
    -   Performed hyperparameter tuning on the best-performing model (Logistic Regression) to optimize its performance.

5.  **Web Application Development:**
    -   Built an interactive web interface with **Streamlit**.
    -   Integrated the trained model to provide real-time sentiment predictions.

6.  **Deployment:**
    -   Deployed the application to **Streamlit Community Cloud** for public access.

## 📊 Model Performance

The final model, an optimized **Logistic Regression** classifier, achieved the following performance on the test set:

| Metric      | Score   |
|-------------|---------|
| Accuracy    | 0.8912  |
| Precision   | 0.8826  |
| Recall      | 0.9043  |
| F1-Score    | 0.8934  |

### Limitations

While the model performs well on straightforward reviews, it can struggle with reviews containing:
-   **Mixed Sentiment:** Sentences with both positive and negative clauses.
-   **Sarcasm and Subtext:** Nuances that require deeper contextual understanding.

Future improvements could involve using more advanced transformer-based models like BERT to better capture these complexities.

## 🚀 How to Run Locally

Follow these steps to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aryan578765/MOVIE-REVIEW-SENTIMENT-ANALYSIS.git
    cd MOVIE-REVIEW-SENTIMENT-ANALYSIS
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

Once the installation is complete, run the Streamlit app with the following command:

```bash
streamlit run app.py
```

The application will open in your web browser, typically at `http://localhost:8501`.

## 📂 Project Structure

```
MOVIE-REVIEW-SENTIMENT-ANALYSIS/
├── app.py                     # Main Streamlit application
├── optimized_model.pkl        # Serialized trained model
├── tfidf_vectorizer.pkl       # Serialized TF-IDF vectorizer
├── requirements.txt           # List of Python dependencies
├── .gitignore                 # Specifies files for Git to ignore
├── README.md                  # This file
└── (Other generated plots/files)
```

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 👨‍💻 Author

**Patel Aryan**

-   GitHub: [aryan578765](https://github.com/aryan578765)

---