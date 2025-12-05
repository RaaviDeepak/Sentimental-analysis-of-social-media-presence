# Sentimental-analysis-of-social-media-presence

📌 Project Overview
This POC demonstrates a full Machine Learning pipeline for analyzing user-generated content from platforms like Twitter, Instagram, and Facebook.
It includes text preprocessing, feature engineering, model training, evaluation, and interpretation through ROC-AUC and high-confidence predictions.

The goal is to build a scalable and transparent sentiment classification system suitable for academic, enterprise, or research applications.

✨ Key Features
Clean text preprocessing & TF-IDF vectorization
Handles class imbalance using Random Oversampling
Multiple ML models:
        Logistic Regression
        Random Forest
        Optional SVC (probability calibrated)
Evaluation Metrics:
        Train/Test Accuracy
        Precision, Recall, F1-Score
        Confusion Matrix
ROC Curves & AUC
Best threshold selection using Youden’s Index
High-confidence sample neutral predictions (≥ 85% probability)
Optional: Sarcasm, Emoji polarity, Negation shift, and other advanced unique features

🧠 Workflow
Load dataset
Clean & preprocess text
Convert text to TF-IDF vectors
Oversample minority sentiment classes
Train ML models
Evaluate model performance
Generate ROC-AUC curves
Display high-confidence predictions
Export final models & visualizations

📥 Installation
pip install numpy pandas scikit-learn imbalanced-learn matplotlib
pip install vaderSentiment emoji textblob

▶️ Usage
Run the main notebook (Google Colab recommended):
  python sentiment_analysis_poc.ipynb
Or execute the script:
  python sentiment_model.py

📊 Output Examples
Accuracy Report
Confusion Matrix
ROC Curves for all classes
AUC Scores
High-confidence predictions
Sample predictions with probability
