import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os


max_rel_dist = 1.1

model = SentenceTransformer('all-MiniLM-L6-v2')

dct = json.load(open("pet_vectors.json", "r"))
dct = {x: np.array(y) for x, y in dct.items()}

while True:
    print("#### Please insert a query to be executed against the documents: ####")
    query = input()

    vector = model.encode(query)
    distances = []

    for doc, docvect in dct.items():
        dist = np.linalg.norm(vector - docvect)

        distances.append([doc, float(dist)])

    distances = sorted(distances, key=lambda x: (x[1], x[0]))

    best = distances[0][1]

    print("\n\nPrompt:\n")
    print("I will provide some contextual information and then an user question. You should not answer until the user question is formulated.")

    for i in range(len(distances)):
        if distances[i][1] / best < max_rel_dist:
            print("\nContextual information "+str(i+1)+":\n"+open(os.path.join("pet_topical", distances[i][0]), "r").read())
            input("## Press any key to continue ## ")

    print("\nUser query: "+str(query))
    input("\n\n## Press any key to continue ##\n\n")

    print("\n\n\n")
