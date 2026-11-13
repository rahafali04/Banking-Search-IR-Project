"""
Main Console Interface - Banking Search IR Project (Phase 1)
Aligned with Dr. Manar Mizher's Student Guide requirements.
"""
from inverted_index import load_clean_csv, build_inverted_index
from boolean_search import BooleanSearch
from ranked_search import RankedSearch

def main():
    dataset_path = "dataset/documents.csv"
    
    # 1. Load dataset & build index automatically in the background
    docs, titles = load_clean_csv(dataset_path)
    index = build_inverted_index(docs)
    
    # Initialize the search engines
    boolean_engine = BooleanSearch(index, docs.keys())
    ranked_engine = RankedSearch(dataset_path)

    while True:
        print("\n" + "=" * 50)
        print("   Banking Support Search (EduSearch / IR Project)")
        print("=" * 50)
        print("1. Boolean Search (AND / OR / NOT)")
        print("2. Ranked Search (TF-IDF & Cosine Similarity)")
        print("3. Exit")
        print("-" * 50)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            q = input("Enter Boolean query: ").strip()
            res = boolean_engine.search(q)
            
            print(f"\nMatching Documents ({len(res)}):")
            if not res:
                print("  No matching documents found.")
            else:
                for doc_id in sorted(res):
                    title_text = titles.get(doc_id, "No Title")
                    print(f"  - D{doc_id:03d}: {title_text}")
                    
        elif choice == "2":
            q = input("Enter query: ").strip()
            results = ranked_engine.search(q, top_k=3)
            
            print(f"\nTop-3 Ranked Results (TF-IDF + Cosine):")
            if not results:
                print("  No relevant documents found.")
            else:
                for res in results:
                    print(f"{res['rank']}. D{res['doc_id']:03d} ({res['score']:.2f}): {res['title']}")
                    
        elif choice == "3":
            print("\nExiting project interface. Good luck with the presentation!")
            break
        else:
            print("\n[!] Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()