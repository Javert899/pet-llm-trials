from sentence_transformers import SentenceTransformer
import os
import json

model = SentenceTransformer('all-MiniLM-L6-v2')
dct = {}

for doc in os.listdir("pet_topical"):
    F = open(os.path.join("pet_topical", doc), "r")
    content = F.read()
    vector = model.encode(content).tolist()
    dct[doc] = vector
    print(doc, vector)
    F.close()

json.dump(dct, open("pet_vectors.json", "w"))
