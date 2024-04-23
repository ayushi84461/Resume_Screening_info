import pickle
import re
from PyPDF2 import PdfReader
from docx import Document

# Load the TF-IDF vectorizer
with open('tfidf.pkl', 'rb') as f:
    loaded_tfidf = pickle.load(f)

# Load the classifier
with open('clf.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)

# Define function to clean resume text
def clean_text(text):
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text

# Define the main function to classify the resume
def classify_resume(uploaded_file):
    # Initialize text variable
    text = ""

    # Check file type and read content
    if uploaded_file.endswith('.pdf'):
        # Read PDF file
        with open(uploaded_file, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
    elif uploaded_file.endswith('.docx'):
        # Read DOCX file
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + '\n'

    # Clean the text
    cleaned_text = clean_text(text)

    # Transform the cleaned text using the loaded TF-IDF vectorizer
    resume_tfidf = loaded_tfidf.transform([cleaned_text])

    # Predict the category of the resume using the loaded classifier
    predicted_category = loaded_clf.predict(resume_tfidf)[0]

    # Print the predicted category
    print("Predicted Category:", predicted_category)

    return predicted_category

# Example usage
classified_category = classify_resume(r"C:\Users\mehvi\OneDrive\Documents\GitHub\resume_classifier\Resume_mehvish.pdf")

print("Predicted Category:", classified_category)
