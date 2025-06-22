# Sentiment Analysis on Movie Reviews

## Overview
This project applies Natural Language Processing (NLP) techniques to analyze and classify the sentiment of movie reviews. Using the [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data), I developed multiple classification models to determine whether a given review expresses a **negative**, **somewhat negative**, **neutral**, **somewhat positive**, or **positive** sentiment.

### Goals
- Preprocess raw text data for NLP modeling
- Extract features using Bag of Words and TF-IDF
- Train classifiers (Logistic Regression, Naive Bayes)
- Evaluate and compare model performance

---

## Dataset
- **Source:** [Kaggle Movie Review Sentiment Dataset](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data)
- **Total Phrases:** 156,060
- **Labels:** Negative, Somewhat Negative, Neutral, Somewhat Positive, Positive

---

## Notebooks
- `Final Project.ipynb`: Full model pipeline including preprocessing, vectorization, training, and evaluation.
- `Kcomp.ipynb`: Model tuning and feature experimentation.
- `nlp_project_report.ipynb`: Summarized walkthrough and report with key outputs and charts.

---

## Methods Used
- Text preprocessing (tokenization, stop word removal, lemmatization)
- Feature extraction: CountVectorizer, TF-IDF
- Classifiers:
  - Multinomial Naive Bayes
  - Logistic Regression
- Evaluation:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix

---

## Results
- Logistic Regression with TF-IDF provided the best performance on the validation set.
- Fine-tuned models demonstrated strong classification performance across all five sentiment categories.
- Visualized key findings and confusion matrices to analyze misclassification trends.

---

## Contributions
- **Dawryn Rosario**  
  Email: darosari@syr.edu  
  GitHub: [https://github.com/darosari](https://github.com/darosari)

---

## Future Work
- Incorporate deep learning models (e.g., LSTM, BERT)
- Improve sentiment granularity through domain-specific embeddings
- Deploy a lightweight web interface for real-time sentiment predictions

---

## License
This project is for educational purposes and is not licensed for commercial use.
