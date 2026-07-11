import nltk
from nltk.tokenize import word_tokenize

# Download only if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = "Yes Bank's management is undergoing a significant transition."

# Step 1: Tokenization
tokens = word_tokenize(text)

# Step 2: POS Tagging
pos_tags = nltk.pos_tag(tokens)

print("Tokens:", tokens)
print("POS Tags:", pos_tags)
