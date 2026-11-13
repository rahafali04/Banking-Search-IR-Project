
# ==========================================
# BANKING77 - Dataset & Preprocessing
# ==========================================

# Step 1: Import libraries

from datasets import load_dataset
import pandas as pd
import nltk
import os

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Step 2: Load BANKING77 dataset

dataset = load_dataset("PolyAI/banking77")

train_data = dataset["train"]
all_data = pd.DataFrame(train_data)
# Check target keywords in the original dataset

keywords = [
    "lost",
    "stolen",
    "replace",
    "replacement",
    "card"
]

print("\nKeyword frequency in original dataset:")

for word in keywords:
    count = all_data["text"].str.contains(
        rf"\b{word}\b",
        case=False,
        na=False
    ).sum()

    print(word, ":", count)


# Display documents containing replace

print("\nDocuments containing replace:")

replace_docs = all_data[
    all_data["text"].str.contains(
        r"\b(replace|replacement|replacing)\b",
        case=False,
        na=False
    )
]

print(replace_docs[["text", "label"]].head(20).to_string(index=False))

print("Total original records:", len(all_data))


# Step 3: Select 8 records from each category

balanced_data = all_data.groupby(
    "label",
    group_keys=False
).sample(
    n=8,
    random_state=42
)

print("Balanced records:", len(balanced_data))


# Step 4: Find documents related to lost or replacement cards

has_card = all_data["text"].str.contains(
    r"\bcard\b",
    case=False,
    na=False
)

has_related_words = all_data["text"].str.contains(
    r"\b(lost|lose|losing|stolen|steal|replace|replacement|replacing|missing)\b",
    case=False,
    na=False
)

targeted_data = all_data[
    has_card & has_related_words
]

print("Relevant card records:", len(targeted_data))


# Step 5: Combine balanced and targeted records

df = pd.concat(
    [balanced_data, targeted_data],
    ignore_index=True
)

# Remove duplicate texts

df = df.drop_duplicates(
    subset=["text"]
)


# Step 6: Keep dataset between 500 and 1000 records

if len(df) > 1000:

    if len(targeted_data) >= 1000:

        df = targeted_data.drop_duplicates(
            subset=["text"]
        ).sample(
            n=1000,
            random_state=42
        )

    else:

        other_data = df[
            ~df["text"].isin(targeted_data["text"])
        ]

        remaining = 1000 - len(targeted_data)

        df = pd.concat([
            targeted_data.drop_duplicates(subset=["text"]),
            other_data.sample(
                n=remaining,
                random_state=42
            )
        ])

df = df.reset_index(drop=True)


# Display selection results

print("\nNumber of selected records:", len(df))

print(
    "Number of categories:",
    df["label"].nunique()
)


# Step 7: Check missing values

print("\nMissing values:")
print(df.isnull().sum())


# Step 8: Check duplicate texts

print("\nDuplicate texts:")
print(df.duplicated(subset=["text"]).sum())


# Step 9: Remove missing and empty texts

df = df.dropna(subset=["text"])

df = df[
    df["text"].str.strip() != ""
]


# Step 10: Remove duplicate texts

df = df.drop_duplicates(
    subset=["text"]
)


print("\nRecords after cleaning:", len(df))


# Step 11: Create document IDs

df = df.reset_index(drop=True)

df["doc_id"] = [
    "DOC" + str(i + 1)
    for i in range(len(df))
]


# Step 12: Create titles

df["title"] = df["text"]


# Step 13: Convert label numbers into category names

category_names = dataset["train"].features["label"].names

df = df.rename(
    columns={"label": "category"}
)

df["category"] = df["category"].apply(
    lambda label: category_names[label]
)


print("\nCategory names:")
print(df[["doc_id", "category"]].head())

print(
    "\nNumber of different categories:",
    df["category"].nunique()
)


# Step 14: Display relevant card documents

print("\nRelevant documents:")

relevant_docs = df[
    df["text"].str.contains(
        r"\b(lost|lose|losing|stolen|steal|replace|replacement|replacing|missing)\b",
        case=False,
        na=False
    )
]

print(
    relevant_docs[
        ["doc_id", "category", "text"]
    ].head(15).to_string(index=False)
)


# Step 15: Download English stopwords

nltk.download("stopwords")


# Step 16: Preprocessing settings

REMOVE_STOPWORDS = True

tokenizer = RegexpTokenizer(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"
)

stop_words = set(
    stopwords.words("english")
)

# Keep negation words because they affect meaning

stop_words = stop_words - {
    "no",
    "not",
    "nor"
}

stemmer = PorterStemmer()


# Step 17: Text preprocessing function

def preprocess_text(text):

    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = tokenizer.tokenize(text)

    # Stop Words Removal
    if REMOVE_STOPWORDS:
        tokens = [
            word for word in tokens
            if word not in stop_words
        ]

    # Stemming
    processed_terms = [
        stemmer.stem(word)
        for word in tokens
    ]

    return " ".join(processed_terms)


# Step 18: Apply preprocessing

df["processed_text"] = df["text"].apply(
    preprocess_text
)


# Step 19: Display preprocessing example

print("\nPreprocessing Example:")

print("\nOriginal Text:")
print(df.iloc[0]["text"])

print("\nProcessed Text:")
print(df.iloc[0]["processed_text"])


# Step 20: Select final document columns

documents = df[
    [
        "doc_id",
        "title",
        "text",
        "category",
        "processed_text"
    ]
]


# Step 21: Create dataset folder

os.makedirs(
    "dataset",
    exist_ok=True
)


# Step 22: Save processed documents

documents.to_csv(
    "dataset/documents.csv",
    index=False,
    encoding="utf-8-sig"
)


# Step 23: Final verification

print("\n================================")
print("Dataset saved successfully!")
print("================================")

print("Total documents:", len(documents))

print(
    "Total categories:",
    documents["category"].nunique()
)

print(
    "Missing values:",
    documents.isnull().sum().sum()
)

print(
    "Duplicate texts:",
    documents["text"].duplicated().sum()
)

print(
    "Empty processed texts:",
    (documents["processed_text"].str.strip() == "").sum()
)

print("Columns:", documents.columns.tolist())

print("Saved file: dataset/documents.csv")
print("\nChecking replace keywords:")

for word in ["lost", "stolen", "replace", "replacement", "card"]:
    count = all_data["text"].str.contains(
        rf"\b{word}\b",
        case=False,
        na=False
    ).sum()

    print(f"{word}: {count}")
    # Check replace documents in selected dataset

print("\nDocuments containing replace or replacement:")

replace_docs = df[
    df["text"].str.contains(
        r"\b(?:replace|replacement|replacing)\b",
        case=False,
        na=False
    )
]

print("Number of replace documents:", len(replace_docs))

print(
    replace_docs[
        ["doc_id", "category", "text", "processed_text"]
    ].to_string(index=False)
)
