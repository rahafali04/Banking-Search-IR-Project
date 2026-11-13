"""
Step 4 test script: one query per operator (AND / OR / NOT).
- Without arguments: runs on a tiny built-in sample and checks results with asserts.
- With a CSV path:   runs the same queries on the real dataset and prints IDs + titles.

Run:  python test_boolean.py
      python test_boolean.py dataset/documents.csv
"""
import sys
from inverted_index import load_clean_csv, build_inverted_index
from boolean_search import BooleanSearch, preprocess_text, show

SAMPLE = {
    1: "I lost my card",
    2: "My card was stolen",
    3: "How do I replace my card",
    4: "My transfer failed",
    5: "I lost my phone",
}

QUERIES = [
    ("AND", "lost AND card"),
    ("OR", "lost OR stolen"),
    ("NOT", "card AND NOT lost"),
    ("Mixed", "(lost OR stolen) AND card"),
]


def run(docs, titles):
    index = build_inverted_index(docs)
    engine = BooleanSearch(index, docs.keys())
    out = {}
    for label, q in QUERIES:
        print(f"\n[{label}] {q}")
        res = engine.search(q)
        show(res, titles)
        out[q] = res
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        docs, titles = load_clean_csv(sys.argv[1])
        run(docs, titles)
    else:
        docs = {i: preprocess_text(t) for i, t in SAMPLE.items()}
        out = run(docs, SAMPLE)
        # Expected results for the sample (assumes the same stem for card/lost)
        assert out["lost AND card"] == {1}
        assert out["lost OR stolen"] == {1, 2, 5}
        assert out["card AND NOT lost"] == {2, 3}
        assert out["(lost OR stolen) AND card"] == {1, 2}
        print("\nAll tests passed.")