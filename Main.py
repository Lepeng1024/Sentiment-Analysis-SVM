from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

import pandas as pd

# Load the data
file_path = r'C:\Users\10415\Desktop\ECE503Proj\data_all.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Example training data (replace with your actual data)
X_train = data['cleaned_content']   
y_train = data['rating']

# split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# MODEL 1 Initialize the components of the pipeline
# tfidf = TfidfVectorizer(max_features=3000)
# svd = TruncatedSVD(n_components = 600) 
# classifier = LogisticRegression(solver='lbfgs', penalty='l2')

# MODEL 2 Initialize the components of the pipeline
# tfidf = TfidfVectorizer(max_features=3000)
# svd = TruncatedSVD(n_components = 600)  
# classifier = MLPClassifier(hidden_layer_sizes = (400),solver='lbfgs', max_iter=500, alpha=10)

# MODEL 3 pipeline for svm
tfidf = TfidfVectorizer(max_features=3000)
# standardize the data
#scaler = StandardScaler()
svd = TruncatedSVD(n_components = 600)  
classifier = LinearSVC()

#-------------
# Create the pipeline
pipeline = make_pipeline(tfidf, svd, classifier)

# time the pipeline
import time
start = time.time()

# Train the pipeline on your data
pipeline.fit(X_train, y_train)

end = time.time()
print("Time:", end - start)

# see how much variance explained

# # Access the explained_variance_ratio_ attribute
# explained_variance = pipeline.named_steps['truncatedsvd'].explained_variance_ratio_

# # Print the total variance explained by all components
# print('Total variance explained by all components:', explained_variance.sum())

# Save the trained pipeline
joblib.dump(pipeline, 'text_processing_pipeline.joblib')

# load 
pipeline = joblib.load('text_processing_pipeline.joblib')

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Print the overall accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_test:", accuracy_score(y_test, y_pred))

# predict on training set and print accuracy
y_pred_train = pipeline.predict(X_train)
print("Accuracy_train:", accuracy_score(y_train, y_pred_train))


# Function to clean and preprocess text
def preprocess_text(text):
    # Lowercase, remove HTML tags, non-alphabets, and numbers
    text = re.sub(r'<.*?>', '', text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization and stopword removal
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)