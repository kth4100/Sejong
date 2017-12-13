from nltk.stem import WordNetLemmatizer

input_words = ['sleeping', 'bottles', 'might', 'washed', 'rabbit', 'localization', 'probably', 'patient', 'orange', 'would', 'dangerous', 'case']

lemmatizer = WordNetLemmatizer()

lemmatizer_names = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)

for word in input_words:
    output = [word,lemmatizer.lemmatize(word,pos='n'),
              lemmatizer.lemmatize(word,pos='v')]
    print(formatted_text.format(*output))
