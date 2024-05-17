import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Advanced preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to handle negations
def handle_negations(text):
    negation_words = set(["not", "no", "never", "n't"])
    words = text.split()
    negated_text = []
    negate = False
    for word in words:
        if word in negation_words:
            negate = True
            negated_text.append(word)
        elif negate:
            negated_text.append(word + "_NEG")
            negate = False
        else:
            negated_text.append(word)
    return ' '.join(negated_text)

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Handle negations
    text = handle_negations(text)
    # Tokenize and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Splitting the data into features and target
X = df['cleaned_review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Define hyperparameters for GridSearchCV
parameters = {
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__min_df': [1, 5],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

# Train the model
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best CV Score: {best_score}')

# Make predictions on the test data
y_pred = grid_search.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Function to classify a new review
def classify_review(review):
    cleaned_review = preprocess_text(review)
    prediction = grid_search.predict([cleaned_review])
    return 'positive' if prediction == 1 else 'negative'

# Interactive loop to classify user input reviews
while True:
    new_review = input('Write your review: ')
    print(f'Review: "{new_review}" is {classify_review(new_review)}')
