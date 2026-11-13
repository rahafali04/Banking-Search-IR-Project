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
| Team Member | Assigned Module | Key Contributions |
| :--- | :--- | :--- |
| **Aya** | Data Preprocessing (`preprocessing.py`) | Downloaded dataset, handled data cleaning, tokenization, stopword removal, stemming, and saved the clean CSV file. |
| **Ghada** | Inverted Index & Boolean Search (`inverted_index.py`, `boolean_search.py`, `test_boolean.py`) | Constructed the dictionary-set inverted index, implemented Boolean operators (AND, OR, NOT), and automated testing. |
| **Rahaf Hamdi** | Dataset Validation & Reporting | Assisted with dataset restructuring, testing query structures, and project documentation. |
| **Rahaf Ali Awwad** | Ranked Search & Integration (`ranked_search.py`, `main.py`) | Implemented TF-IDF vectorization, Cosine Similarity ranking, top-k retrieval, and the interactive console interface. |

---

## 🛠️ Project Structure
```text
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