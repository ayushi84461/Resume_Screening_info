import streamlit as st
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

# List of class names
classes = ['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing',
           'Mechanical Engineer', 'Sales', 'Health and fitness',
           'Civil Engineer', 'Java Developer', 'Business Analyst',
           'SAP Developer', 'Automation Testing', 'Electrical Engineering',
           'Operations Manager', 'Python Developer', 'DevOps Engineer',
           'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
           'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']

# Example mapping of category integers to class names
category_mapping = {
    # array([ 6, 12,  0,  1, 24, 16, 22, 14,  5, 15,  4, 21,  2, 11, 18, 20,  8,
    #    17, 19,  7, 13, 10,  9,  3, 23])
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and fitness',
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain',
    23: 'Testing'
}

# Define the main function to classify the resume
def classify_resume(uploaded_file):
    # Initialize text variable
    text = ""

    # Check file type and read content
    if uploaded_file.type == 'application/pdf':
        # Read PDF file
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
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

    # Get the class name from the category integer
    class_name = category_mapping[predicted_category]

    return predicted_category, class_name

# Streamlit UI
def main():
    st.title("Resume Classifier")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload a resume (PDF or Word)", type=['pdf', 'docx'])

    if uploaded_file is not None:
        # Display the uploaded file
        st.write("Uploaded resume:")
        st.write(uploaded_file)

        # Classify the resume
        category_int, class_name = classify_resume(uploaded_file)

        # Display the classification result
        st.write("Classification result")
        st.write("Predicted Category:", category_int)
        st.write("Predicted Class:", class_name)

if __name__ == "__main__":
    main()
