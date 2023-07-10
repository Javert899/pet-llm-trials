# Required Libraries
import nltk
from nltk.corpus import brown
from nltk.tokenize import TextTilingTokenizer
import os

# Initialize TextTilingTokenizer
tt = TextTilingTokenizer()

for docname in os.listdir("pet"):
    # Ingest Document
    F = open(os.path.join("pet", docname), "r")
    document = F.read()
    F.close()

    if len(document) < 10000:
        target_name = os.path.join("pet_topical", docname)

        document = document.replace("\n\n", "\n").strip()
        print(docname, len(document))

        F = open(target_name, "w")
        F.write(document)
        F.close()

    else:
        # Tokenizing document into sentences
        sentences = nltk.sent_tokenize(document)

        # Break down text into topical sections
        try:
            tiles = tt.tokenize(document)

            # Print each tile
            for i, tile in enumerate(tiles):
                tile = tile.replace("\n\n", "\n").strip()
                target_name = os.path.join("pet_topical", docname + "_sec_"+str(i+1))

                F = open(target_name, "w")
                F.write(tile)
                F.close()

                #print(f"Section {i+1}:\n{tile}\n")

        except:
            target_name = os.path.join("pet_topical", docname)

            document = document.replace("\n\n", "\n").strip()

            F = open(target_name, "w")
            F.write(document)
            F.close()
