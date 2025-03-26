# 🔧 Install required packages
!pip install -q transformers sentence-transformers fuzzywuzzy[speedup] faiss-cpu pandas
# Facebook AI Similarity Search (FAISS)

# Imports
import pandas as pd
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import torch
import json

# Step 1: Sample input (can be structured or free-text descriptions)
transactions = [
    {"id": 1, "description": "Transfer to ShellCorp Intl in Cayman Islands."},
    {"id": 2, "description": "Donation sent to GlobalUnity Non-Profit Org."},
    {"id": 3, "description": "Payment made to XBridge Ventures, previously flagged by compliance."},
    {"id": 4, "description": "Wire transfer to Oceanic Holdings in Singapore."},
    {"id": 5, "description": "Contribution received by HelpingHands Charity Foundation."},
    {"id": 6, "description": "Settlement payment processed for MetroFinance Inc, under review."},
    {"id": 7, "description": "Invoice payment to GlobalTrade Partners Ltd in Dubai."},
    {"id": 8, "description": "Refund issued by RetailCentral Corp for overcharge."},
    {"id": 9, "description": "Reimbursement from BlueChip LLC for travel expenses."},
    {"id": 10, "description": "Payment processed to SecureInvestments, pending further verification."}
]


df = pd.DataFrame(transactions)

# Step 2: Load NER model (HF)
ner_pipeline = pipeline("ner", grouped_entities=True)

# Step 3: Named Entity Extraction
df["entities"] = df["description"].apply(lambda x: [ent['word'] for ent in ner_pipeline(x)])

# Step 4: Create embedding model + index of known safe/flagged entities
known_entities = ["Shell Corporation International", "UNICEF", "World Bank", "XBridge Ventures", "GlobalUnity Foundation"]
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
known_embeds = embed_model.encode(known_entities)

index = faiss.IndexFlatL2(known_embeds.shape[1])
index.add(known_embeds)

# Step 5: Verify and risk score
def verify_and_score(extracted_entities):
    result = []
    risk_keywords = ['shell', 'offshore', 'flagged', 'cayman']

    for entity in extracted_entities:
        embed = embed_model.encode([entity])
        _, I = index.search(embed, k=1)
        match = known_entities[I[0][0]]
        fuzz_score = fuzz.partial_ratio(entity.lower(), match.lower())

        # Heuristic risk scoring
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

df["risk_eval"] = df["entities"].apply(verify_and_score)

# Step 6: Generate LLM-based justifications using FLAN-T5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

llm = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(llm)
model = AutoModelForSeq2SeqLM.from_pretrained(llm)

def justify(entity, match, score):
    prompt = f"""
    You are an expert risk analysis assistant.

    Analyze the entity: "{entity}"
    It matched to: "{match}"
    It has been assigned a risk score of {score}.

    Briefly explain why this entity might carry that level of risk.
    """

    inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,  # deterministic output
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Apply justification per entity
justifications = []
for row in df.itertuples():
    reasons = []
    for item in row.risk_eval:
        reason = justify(item['entity'], item['matched_to'], item['risk_score'])
        item['justification'] = reason
        reasons.append(item)
    justifications.append(reasons)

df["risk_eval"] = justifications

# Step 7: Output final result
print("Final Risk Analysis Output (JSON-like):\n")
for row in df.itertuples():
    output = {
        "transaction_id": row.id,
        "description": row.description,
        "entities": row.risk_eval
    }
    print(json.dumps(output, indent=2))

# 🔗 OpenCorporates API lookup function
# This has been commented as API token request was rejected.
# import requests

# def lookup_entity_opencorp(name, jurisdiction_code=""):
#     api_key = "YOUR_API_KEY"  #Replace this with your actual API token
#     base_url = "https://api.opencorporates.com/v0.4/companies/search"

#     params = {
#         "q": name,
#         "api_token": api_key
#     }
#     if jurisdiction_code:
#         params["jurisdiction_code"] = jurisdiction_code

#     try:
#         response = requests.get(base_url, params=params)
#         data = response.json()
#         if "results" in data and data["results"]["companies"]:
#             top_result = data["results"]["companies"][0]["company"]
#             return {
#                 "name": top_result.get("name"),
#                 "jurisdiction": top_result.get("jurisdiction_code"),
#                 "company_number": top_result.get("company_number"),
#                 "status": top_result.get("current_status"),
#                 "source_url": top_result.get("opencorporates_url")
#             }
#         else:
#             return {"error": "No match found"}
#     except Exception as e:
#         return {"error": str(e)}

#  Phase 2 integration: Add LLM justification + OpenCorporates verification
# justifications = []
# for row in df.itertuples():
#     enriched_entities = []
#     for item in row.risk_eval:
#         # LLM-based justification
#         reason = justify(item['entity'], item['matched_to'], item['risk_score'])
#         item['justification'] = reason

#         #  Real-time entity verification using OpenCorporates
#         api_result = lookup_entity_opencorp(item['entity'])
#         item['open_corporates'] = api_result  # Attach API enrichment

#         enriched_entities.append(item)
#     justifications.append(enriched_entities)

# # Update DataFrame with enriched info
# df["risk_eval"] = justifications

# # Final JSON-like output
# import json
# print("Enriched Risk Analysis Output with OpenCorporates:\n")
# for row in df.itertuples():
#     output = {
#         "transaction_id": row.id,
#         "description": row.description,
#         "entities": row.risk_eval
#     }
#     print(json.dumps(output, indent=2))