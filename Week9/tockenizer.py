Python 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 18:41:36) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> from nltk.tokenize import sent_tokenize, \
        word_tokenize, WordPunctTokenizer

# Define input text
input_text = "Megafires, individual fires that burn more than 100,000 acres, are on the rise in the western United States -- the direct result of unintentional yet massive changes we've brought to the forests through a century of misguided management. What steps can we take to avoid further destruction? Forest ecologist Paul Hessburg confronts some tough truths about wildfires and details how we can help restore the natural balance of the landscape."

# Sentence tokenizer 
print("\nSentence tokenizer:")
print(sent_tokenize(input_text))

# Word tokenizer
print("\nWord tokenizer:")
print(word_tokenize(input_text))

# WordPunct tokenizer
print("\nWord punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))
