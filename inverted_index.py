"""
Person 2 - Inverted Index Module
Builds a term -> set(doc_id) inverted index (Python dict + sets)
from the CLEAN dataset produced by Person 1.

Expected clean CSV columns:
    doc_id, title, text, processed_text
where processed_text = final terms (after lowercase, tokenize,
stopwords, stemming) separated by spaces.
"""
import csv
import sys


def load_clean_csv(path="dataset/documents.csv"):
    """Return (docs, titles): docs = {doc_id: [terms]}, titles = {doc_id: title}."""
    docs, titles = {}, {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            doc_id = int(row["doc_id"].replace("DOC", ""))
            docs[doc_id] = row["processed_text"].split()
            titles[doc_id] = row["title"]
    return docs, titles


def build_inverted_index(docs):
    """docs: {doc_id: [terms]} -> index: {term: {doc_ids}}"""
    index = {}
    for doc_id, terms in docs.items():
        for term in terms:
            index.setdefault(term, set()).add(doc_id)
    return index


def show_postings(index, terms=None, limit=10):
    """Print the postings list (term -> doc IDs)."""
    if terms is None:
        terms = sorted(index)[:limit]
    for t in terms:
        print(f"{t:<15} -> {sorted(index.get(t, set()))}")


def index_stats(index, docs):
    print(f"Documents indexed : {len(docs)}")
    print(f"Unique terms      : {len(index)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "clean_banking77.csv"
    docs, titles = load_clean_csv(path)
    index = build_inverted_index(docs)
    index_stats(index, docs)
    print("\n--- Sample postings ---")
    show_postings(index, limit=10)
    print("\n--- Postings for query words ---")
    show_postings(index, ["lost", "card", "replac"])
