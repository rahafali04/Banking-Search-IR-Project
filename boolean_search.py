"""
Person 3 - Boolean Search Module
AND = intersection, OR = union, NOT = complement (vs. ALL indexed docs).
Precedence: NOT > AND > OR. Parentheses supported.
Adjacent words without an operator are treated as AND.

Run:  python boolean_search.py clean_banking77.csv
"""
import re
import sys
from inverted_index import load_clean_csv, build_inverted_index

# Use Person 1's preprocessing so query terms match index terms.
try:
    from preprocessing import preprocess_text      # returns list of final terms
except ImportError:
    from nltk.stem import PorterStemmer             # fallback (pip install nltk)
    _stem = PorterStemmer().stem

    def preprocess_text(text):
        return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]

OPERATORS = {"AND", "OR", "NOT"}


class BooleanSearch:
    def __init__(self, index, all_ids):
        self.index = index
        self.all_ids = set(all_ids)     # universe for NOT

    # ---------- tokenizer ----------
    @staticmethod
    def _tokenize(query):
        return re.findall(r"\(|\)|[^\s()]+", query)

    # ---------- recursive-descent parser ----------
    def search(self, query):
        self.toks, self.pos = self._tokenize(query), 0
        result = self._or()
        if self.pos != len(self.toks):
            raise ValueError("Malformed query")
        return result

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _or(self):
        res = self._and()
        while self._peek() == "OR":
            self.pos += 1
            res = res | self._and()                 # union
        return res

    def _and(self):
        res = self._not()
        while self._peek() is not None and self._peek() not in ("OR", ")"):
            if self._peek() == "AND":
                self.pos += 1                       # implicit AND otherwise
            res = res & self._not()                 # intersection
        return res

    def _not(self):
        if self._peek() == "NOT":
            self.pos += 1
            return self.all_ids - self._not()       # complement
        return self._atom()

    def _atom(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of query")
        self.pos += 1
        if tok == "(":
            res = self._or()
            if self._peek() != ")":
                raise ValueError("Missing ')'")
            self.pos += 1
            return res
        terms = preprocess_text(tok)
        if not terms:                               # e.g. a stopword
            print(f"  [warning] '{tok}' removed by preprocessing")
            return set()
        return set(self.index.get(terms[0], set()))


def show(results, titles, limit=10):
    print(f"  Matches: {len(results)}")
    for d in sorted(results)[:limit]:
        print(f"   [{d}] {titles[d]}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "clean_banking77.csv"
    docs, titles = load_clean_csv(path)
    index = build_inverted_index(docs)
    engine = BooleanSearch(index, docs.keys())

    print("Boolean search ready. Examples: lost AND card | lost OR stolen | card AND NOT lost")
    print("Type 'exit' to quit.\n")
    while True:
        q = input("Query> ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue
        try:
            show(engine.search(q), titles)
        except ValueError as e:
            print("  Error:", e)
