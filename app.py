import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Title
st.set_page_config(page_title="Resume Screening Tool", layout="wide")
st.title("📄 AI-Powered Resume Screening Tool")

# Upload job description
job_description = st.text_area("📝 Enter Job Description", height=200)

# Upload multiple resumes
st.subheader("📂 Upload Resumes")
uploaded_files = st.file_uploader("Choose resume text files", type=["txt"], accept_multiple_files=True)


# Load resumes into dictionary
def read_resumes(files):
    resumes = {}
    for file in files:
        content = file.read().decode("utf-8")
        resumes[file.name] = content
    return resumes


# TF-IDF + Cosine Similarity Scoring
def rank_resumes(jd, resumes_dict):
    docs = [jd] + list(resumes_dict.values())
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(docs)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Pair scores with filenames
    scored = list(zip(resumes_dict.keys(), similarity_scores))
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return ranked


# When everything is ready
if job_description and uploaded_files:
    resumes_dict = read_resumes(uploaded_files)
    results = rank_resumes(job_description, resumes_dict)

    st.subheader("📊 Ranked Resumes")
    for rank, (filename, score) in enumerate(results, 1):
        st.markdown(f"**{rank}. {filename}** — Score: `{score:.2f}`")
else:
    st.info("Please enter a job description and upload at least one resume.")
