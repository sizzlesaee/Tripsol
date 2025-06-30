# Tripsol
import pandas as pd
!pip install hf_xet
!pip install txtai
!pip install rapidfuzz
!pip install pyspellchecker
import txtai
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
df2 = pd.read_csv(r"C:\Users\study\OneDrive\Documents\indiadata.csv")
df = pd.read_csv(r"C:\Users\study\OneDrive\Documents\Destination,Type.csv")


from txtai.embeddings import Embeddings

types = df["type"].dropna().astype(str).unique().tolist()

embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
embeddings.index(types)

def recommendation(typewise):
    result = embeddings.search(typewise, 1)
    if not result:
            return "No matching category found."

    destinationmatch = types[result[0][0]]
    
    destypes = df[df['type'].str.lower() == destinationmatch.lower()]
    return destypes[['destination']]

destinationinput = input("Enter type of destination: ")
print(recommendation(destinationinput))


from txtai.embeddings import Embeddings
from rapidfuzz import process
#citywise

dest= df2["destination"].dropna().astype(str).unique().tolist()

embeddings= Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
embeddings.index(dest)

def recommend(citywise):
    result = embeddings.search(citywise, 1)
    if not result:
        match, score, _ = process.extractOne(citywise, dest)
        if score > 80:
            city = match
        else:
            return "No matching category found."

    city = dest[result[0][0]]
    print(f"\nAttractions in '{city}'")
  
    places = df2[df2['destination'].str.lower()==city.lower()]
    return places[['attraction', 'entry_fee', 'time_required', 'best_time_to_visit']]

cityinput=input("What place would you like to visit?")

print(recommend(cityinput))
