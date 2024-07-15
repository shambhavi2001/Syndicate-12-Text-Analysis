# News Headline Topic Classification Using SVM and Random Forest Models

## Project Overview

This project aims to develop a text classification model to categorize news article titles into one of eight predefined topics. The project explores the use of Support Vector Machines (SVM) and Random Forest models, incorporating hyperparameter tuning and feature selection to improve classification performance. Additionally, an extension of the project investigates the potential of classifying the articles based on their URLs.

## Project Structure

### 1. Introduction
The objective of this project is to build and evaluate machine learning models for text classification, focusing on news article titles. The project includes:
- Data pre-processing
- Feature extraction
- Feature selection
- Model training and evaluation
- Testing the model on URLs

### 2. Data Pre-processing
- **Tokenization**: Splitting text into individual tokens.
- **Stop Words Removal**: Removing common words that do not contribute to the classification.
- **Lemmatization**: Converting words to their base forms.
- **Normalization**: Converting text to lowercase and removing punctuation.

### 3. Feature Extraction
- **TF-IDF Vectorization**: Converting text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) representation.

### 4. Feature Selection
- **Chi-Square Test**: Selecting the top 850 features based on the chi-square statistical test to reduce dimensionality and improve model performance.

### 5. Model Training
- **Support Vector Machine (SVM)**: Trained on the pre-processed and feature-selected titles using hyperparameter tuning via RandomizedSearchCV.
- **Random Forest**: Hyperparameter tuning ranges provided, but SVM was primarily used in the final implementation.

### 6. URL Classification (Extension)
- **Pre-processing URLs**: Removing "https://www." from the URLs to focus on the meaningful parts.
- **Feature Extraction and Selection**: Similar steps as with the titles, using TF-IDF vectorization and chi-square feature selection.
- **Model Testing on URLs**: The trained SVM model on titles was tested on URLs to evaluate its performance in classifying topics based on URL content.

## Dependencies

- Python 3.7+
- Pandas
- Numpy
- Scikit-learn
- NLTK
- Scipy

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/shambhavi2001/Syndicate-12-Text-Analysis.git]
   cd text-classification
   ```
   
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the root directory:
   - `labelled_newscatcher_dataset.csv`

## Evaluation

- **Classification Report**: Displays precision, recall, F1-score, and support for each class.
- **Balanced Accuracy**: Balanced accuracy considering the imbalanced class distribution.

## Results

- **SVM Model on Titles**:
  - High accuracy and balanced accuracy.
  - Detailed classification report provided in the output.
  - High computation time

- **SVM Model on URLs**:
  - Performance metrics evaluated and compared with title-based classification.
  - High computation time
  - Higher accuracy

- **Random Forest on Titles**
  - Lower accuracy
  - Cheaper computation time

## Contributors 

- **Team**: Syndicate 12
- **Mmebers**: Anuraag Dasari, Shambhavi Gupta, Chamilka Udugoda, Kevin Zhang, Meagan Loh
