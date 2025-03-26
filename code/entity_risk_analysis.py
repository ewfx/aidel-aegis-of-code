# entity_risk_analysis.py

import pandas as pd
import faiss
import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

# -------------------------------
# Load Models
# -------------------------------
ner_pipeline = pipeline("ner", grouped_entities=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(llm)
model = AutoModelForSeq2SeqLM.from_pretrained(llm)

# -------------------------------
# Known Entities Setup
# -------------------------------
known_entities = [
    "Shell Corporation International", "UNICEF", "World Bank",
    "XBridge Ventures", "GlobalUnity Foundation"
]
known_embeds = embed_model.encode(known_entities)
index = faiss.IndexFlatL2(known_embeds.shape[1])
index.add(known_embeds)

# -------------------------------
# Core Functions
# -------------------------------
def extract_entities(text):
    return [ent['word'] for ent in ner_pipeline(text)]

def verify_and_score(entities):
    result = []
    risk_keywords = ['shell', 'offshore', 'flagged', 'cayman']
    for entity in entities:
        embed = embed_model.encode([entity])
        _, I = index.search(embed, k=1)
        match = known_entities[I[0][0]]
        fuzz_score = fuzz.partial_ratio(entity.lower(), match.lower())

        keyword_risk = any(risk in entity.lower() for risk in risk_keywords)
        risk = 70 if keyword_risk else 30
        if fuzz_score < 60:
            risk += 20

        result.append({
            "entity": entity,
            "matched_to": match,
            "match_score": fuzz_score,
            "risk_score": min(risk, 100)
        })
    return result

def generate_justification(entity, match, score):
    prompt = f"""You are an expert risk analysis assistant.
Analyze the entity: "{entity}"
It matched to: "{match}"
It has been assigned a risk score of {score}.
Briefly explain why this entity might carry that level of risk."""
    inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def analyze_text(text):
    entities = extract_entities(text)
    evaluations = verify_and_score(entities)
    for eval in evaluations:
        justification = generate_justification(
            eval['entity'], eval['matched_to'], eval['risk_score']
        )
        eval['justification'] = justification
    return {
        "input_text": text,
        "entities": evaluations
    }

# -------------------------------
# Test Sample
# -------------------------------
if __name__ == "__main__":
    sample_text = "Payment made to XBridge Ventures, previously flagged by compliance."
    result = analyze_text(sample_text)
    print(json.dumps(result, indent=4))
