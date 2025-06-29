# Tripsol
from txtai.embeddings import Embeddings

types = df["type"].dropna().astype(str).unique().tolist()

embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
embeddings.index(types)

def recommendation(typewise):
    result = embeddings.search(typewise)
    if not result:
            return "No matching category found."

    destinationmatch = types[result[0][0]]
    
    destypes = df[df['type'].str.lower() == destinationmatch.lower()]
    return destypes[['destination']]

destinationinput = input("Enter type of destination: ")
print(recommendation(destinationinput))
