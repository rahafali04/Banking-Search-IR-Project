# Banking Search Information Retrieval System (Phase 1)
**Course:** Information Retrieval Systems (902333)[cite: 2]  
**Institution:** Al-Bayt University – Prince Al Hussein Bin Abdullah II Faculty of IT[cite: 2]  
**Project Topic:** Banking Support Search (Banking77 Dataset)[cite: 3]

---

## 📋 Project Overview
This project builds a core small-scale text search engine using the real-world Banking77 dataset[cite: 2, 3]. The system integrates text preprocessing, an inverted index built with Python dictionaries and sets, exact Boolean matching (AND, OR, NOT), and vector space model ranking using TF-IDF and Cosine Similarity[cite: 2, 5].

The goal of Phase 1 is to provide a robust functional console-based search application where users can retrieve relevant banking support passages through both Boolean and Ranked search modalities[cite: 2].

---

## 👥 Team Members & Contributions

| Team Member | Assigned Module | Key Contributions & Video Demonstration Plan |
| :--- | :--- | :--- |
| **Aya** | **Data Preprocessing** <br>(`preprocessing.py`) | • Downloaded the BANKING77 dataset, cleaned text, and applied Lowercasing, Tokenization, Stopwords Removal, and Stemming.<br>• **In Video:** Runs code and demonstrates a live "Before / After" example of text processing. |
| **Ghada** | **Inverted Index** <br>(`inverted_index.py`) | • Built the term-to-document-ID inverted index using Python dictionaries and Sets based on the clean corpus.<br>• **In Video:** Runs the index script and explains the structure of postings. |
| **Rahaf Hamdi** | **Boolean Search** <br>(`boolean_search.py`, `test_boolean.py`) | • Implemented logical operators: AND (intersection), OR (union), and NOT (complement) against the inverted index.<br>• **In Video:** Executes Boolean queries (e.g., `lost AND card`) and shows matching document IDs. |
| **Rahaf Ali Awwad** | **Ranked Search & Integration** <br>(`ranked_search.py`, `main.py`) | • Implemented TF-IDF vectorization, Cosine Similarity ranking, top-k retrieval, and the main integrated console interface (`main.py`).<br>• **In Video:** Runs ranked queries, demonstrates similarity scores, and presents the integrated system. |

---

## 🛠️ Project Structure
```text
Banking-Search-IR-Project/
│
├── dataset/
│   └── documents.csv          # Preprocessed clean banking corpus (743 records)
├── __pycache__/
├── preprocessing.py           # Data extraction and cleaning pipeline (Aya)
├── inverted_index.py          # Term-to-doc_id inverted index module (Ghada)
├── boolean_search.py          # Boolean query parsing and set operations (Rahaf Hamdi)
├── test_boolean.py            # Automated test script for Boolean operators (Rahaf Hamdi)
├── ranked_search.py           # TF-IDF vector space model & Cosine Similarity (Rahaf Ali)
├── main.py                    # Interactive console application / Main integration (Rahaf Ali)
├── requirements.txt           # Project dependencies list
└── README.md                  # Project documentation