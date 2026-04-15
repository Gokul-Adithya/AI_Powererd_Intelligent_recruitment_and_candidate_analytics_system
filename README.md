# 🤖 Resume-Based Candidate Analytics System

An AI-powered recruitment system that automates **resume parsing, skill extraction, and candidate ranking** using NLP and Machine Learning.

---

## 🚀 Overview

This project helps recruiters analyze multiple resumes efficiently by:

- Extracting structured information from resumes (PDF/DOCX)
- Matching candidates with job descriptions using AI
- Ranking candidates based on suitability
- Providing visual analytics and reports

---

## 📁 Project Structure
Resume_based_candidates_analytics_system/<br>
│<br>
├── resumes/ # Folder containing input resumes<br>
├── app2.py # Main Streamlit application<br>
├── requirements.txt # Dependencies<br>
└── README.md # Project documentation<br>


---

## ⚙️ Features

### 📄 Resume Parsing
- Extracts:
  - Name
  - Email
  - Phone Number
  - LinkedIn / GitHub
  - Skills
  - Education
  - Experience
  - Projects
  - Certifications

### 🧠 AI-Based Matching
- Uses **BERT (MiniLM)** embeddings
- Computes **cosine similarity** with job description

### 🎯 Candidate Ranking
- Combines:
  - Semantic similarity (BERT)
  - Logistic Regression prediction
- Generates a **final score (%)**

### 📊 Dashboard & Analytics
- Candidate ranking view
- Skill distribution charts
- Status classification:
  - ✅ Shortlisted
  - ⚠️ Under Review
  - ❌ Rejected

### 📄 Reports
- Export results as CSV
- Generate PDF recruitment reports

---

## 🧠 Tech Stack

- **Frontend**: Streamlit  
- **NLP**: spaCy  
- **Embeddings**: HuggingFace Transformers  
- **ML Model**: Logistic Regression (Scikit-learn)  
- **Data Processing**: NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  
- **File Parsing**: PyPDF2, pdfplumber, python-docx  
- **Report Generation**: ReportLab  

---

## 🔄 System Workflow

1. Upload resumes (PDF/DOCX)
2. Extract text from files
3. Parse resume data
4. Generate BERT embeddings
5. Compare with job description
6. Train ML model
7. Rank candidates
8. Display analytics
9. Generate reports

---

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/Gokul-Adithya/Resume_based_candidates_analytics_system.git

# Navigate to project folder
cd Resume_based_candidates_analytics_system

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt 

## ▶️ Running the Application
streamlit run app2.py
```
## 🧮 Scoring Logic
Final Score = (BERT Similarity + ML Prediction Probability) / 2


- **Similarity Score** → Measures semantic match  
- **Prediction Score** → ML-based suitability  

---

## 📊 Output

- Ranked candidate list  
- Skill gap analysis  
- Visual dashboards  
- CSV download  
- PDF recruitment report  

---

## ⚠️ Limitations

- Skill extraction is keyword-based  
- Resume format variations may affect parsing  
---

## 🔮 Future Improvements

- Improve parsing using LLMs  
- Add recruiter feedback loop  
- Deploy as a web platform  
- Enhance skill matching using embeddings  
