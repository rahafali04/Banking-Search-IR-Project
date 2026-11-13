Banking Search Information Retrieval System (Phase 1)

Course: Information Retrieval Systems (902333)

Institution: Al-Bayt University – Prince Al Hussein Bin Abdullah II Faculty of IT

Project Topic: Banking Support Search (Banking77 Dataset)

📋 Project Overview

This project builds a core small-scale text search engine using the real-world Banking77 dataset. The system integrates text preprocessing, an inverted index built with Python dictionaries and sets, exact Boolean matching (AND, OR, NOT), and vector space model ranking using TF-IDF and Cosine Similarity.

The goal of Phase 1 is to provide a robust functional console-based search application where users can retrieve relevant banking support passages through both Boolean and Ranked search modalities.

👥 Team Contributions

Team Member

Assigned Module

Key Contributions

Member 1

Data Preprocessing (preprocessing.py)

Downloaded dataset, handled data cleaning, tokenization, stopword removal, stemming, and saved the clean CSV file.

Member 2

Inverted Index & Boolean Search (inverted_index.py, boolean_search.py)

Constructed the dictionary-set inverted index and implemented Boolean operators (AND, OR, NOT).

Member 3 (Rahaf)

Ranked Search & Integration (ranked_search.py, main.py)

Implemented TF-IDF vectorization, Cosine Similarity ranking, top-k retrieval, and the interactive console interface.

🛠️ Project Structure

Banking-Search-IR-Project/
│
├── dataset/
│   └── documents.csv          # Preprocessed clean banking corpus (743 records)
├── __pycache__/
├── preprocessing.py           # Data extraction and cleaning pipeline
├── inverted_index.py          # Term-to-doc_id dictionary inverted index module
├── boolean_search.py          # Boolean query parsing and set operations
├── test_boolean.py            # Automated test script for Boolean operators
├── ranked_search.py           # TF-IDF vector space model & Cosine Similarity ranking
├── main.py                    # Interactive console application (Main runner)
├── requirements.txt           # Project dependencies list
└── README.md                  # Project documentation


⚙️ Installation & Setup

Clone or Open the Repository:
Ensure you are in the root directory of the project in your terminal:

cd Banking-Search-IR-Project


Install Required Packages:
Make sure pandas, scikit-learn, nltk, and datasets are installed:

pip install -r requirements.txt


🚀 How to Run the Project

1. Run Data Preprocessing (Generates Clean Corpus)

python preprocessing.py


2. Test Inverted Index & Postings

python inverted_index.py dataset/documents.csv


3. Test Boolean Search (Automated Unit Tests)

python test_boolean.py dataset/documents.csv


4. Run the Full Interactive Console Application (Main Interface)

To launch the interactive system matching the course specifications:

python main.py


In the interactive menu:

Choose 1 to run Boolean Search (e.g., query: lost AND card).

Choose 2 to run Ranked Search (e.g., query: how do I replace my stolen card).

Choose 3 to exit.

📊 Technical Architecture & Verification

Corpus Size: 743 structured banking intent passages across 77 categories.

Boolean Engine: Translates logical operators into set intersections (AND), unions (OR), and complements (NOT).

Ranking Model: Utilizes scikit-learn's TfidfVectorizer and cosine_similarity to calculate vector alignment scores, returning top-k sorted results descendingly.