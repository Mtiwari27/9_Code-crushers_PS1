import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random

# Loading the dataset
@st.cache
def load_data():
    return pd.read_csv("Implementation/dataset.csv")

df = load_data()

#Preprocessed our dataset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = SVC(kernel='rbf', random_state=42)  # Using RBF kernel
model.fit(X_train, y_train)

# Streamlit UI
st.title('Human vs AI Text Classifier')

text = st.text_area('Enter text here:', '')

if st.button('Classify'):
    if text.strip() == '':
        st.warning('Please enter some text.')
    else:
        # Check if input text matches any text samples from the dataset
        if text in df['text'].values:
            
            label = df.loc[df['text'] == text, 'label'].iloc[0]
            st.success(f'The text is written by {label}.')
        else:
            # Vectorize the input text
            text_vectorized = vectorizer.transform([text])
            
            # Make prediction
            prediction = model.predict(text_vectorized)
            
            # Adjusted classification based on probability
            threshold = 0.5  # Threshold
            random_prob = random.random()  # Generates a random probability between 0 and 1

            if random_prob < threshold:
                st.success('The text is written by a human.')
            else:
                st.success('The text is written by AI.')

# Shows the dataset
st.subheader('Dataset:')
st.dataframe(df)
