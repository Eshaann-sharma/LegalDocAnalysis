from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import sent_tokenize

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Clause category prototypes (semantic anchors)
CLAUSE_PROTOTYPES = {
    "Termination": "termination of agreement, expiry of lease, notice period, early termination",
    "Deposit": "security deposit, advance payment, refundable deposit",
    "Rent": "monthly rent, payable amount, rental amount",
    "Usage": "use of property, residential purpose, commercial use",
    "Subletting": "sublet, third party occupation, transfer of possession",
    "Maintenance": "maintenance, repair, cleanliness responsibility",
    "Confidentiality": "confidential information, non-disclosure",
}

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Precompute prototype embeddings
PROTO_EMB = {k: embed(v) for k,v in CLAUSE_PROTOTYPES.items()}

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

def split_into_blocks(text):
    blocks = []
    temp = ""

    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 3:
            if temp:
                blocks.append(temp.strip())
                temp = ""
        else:
            temp += " " + line

    if temp:
        blocks.append(temp.strip())

    return blocks

def detect_clauses(text, threshold=0.6):
    sentences = sent_tokenize(text)
    results = []

    for s in sentences:
        s_emb = embed(s)
        best_match = None
        best_score = 0

        for ctype, p_emb in PROTO_EMB.items():
            sim = cosine_sim(s_emb, p_emb)
            if sim > best_score:
                best_score = sim
                best_match = ctype

        if best_score > threshold:
            results.append(f"[{best_match} | {round(best_score,2)}] {s.strip()}")

    return results
