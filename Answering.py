# NLP-FAQ answering using Bert embeddings

import pandas as pd
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from IPython.display import clear_output

# reading the question and answers
df = pd.read_csv("FAQ.csv")

# processing the sentence
def clean_sentence(sentence, stopwords=False):
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    
    if stopwords:
        sentence = remove_stopwords(sentence)
        
    return sentence

# processing the DataFrame
def get_cleaned_sentences(df, stopwords=False):
    cleaned_sentences=[]
    
    for index, row in df.iterrows():
        cleaned = clean_sentence(row["questions"], stopwords)
        cleaned_sentences.append(cleaned)
        
    return cleaned_sentences

# Prints the most appropriate answer
def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf, sentences, min_similarity):
    max_sim = -1;
    index_sim = -1
    
    # cosine similarity
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        #print(index, sim, sentences[index])
        
        if sim > max_sim:
            max_sim = sim
            index_sim = index
    
    print(f"\nSimilarity: {max_sim}")
    if max_sim > min_similarity:
        print("\nRetrived: ", FAQdf.iloc[index_sim, 0])
        print(FAQdf.iloc[index_sim, 1])
        print("\n")
    else:
        print("\nCouldn't find a relevant answer to your question.\nPlease write us a mail of your query.\nOur experts will reach out to you ASAP.\n")

# Bert model
bert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
#bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Pre-processing the questions before encoding
cleaned_sentences = get_cleaned_sentences(df, stopwords=False)

sent_bertphrase_embeddings=[]

for sent in cleaned_sentences:
    sent_bertphrase_embeddings.append(bert_model.encode([sent]))

# threshold to print an answer
min_similarity = 0.1   

# repeatedly asking for a question from user
while True:
    
    # asking for a question
    print("Please enter your Query or press 'q' to exit.")
    question = input("Question: ")

    clear_output()
    print("Question: " + question)
    
    # break condition
    if question == 'q':
        clear_output()
        print("Have a nice day !")
        break
    
    # preprocessing the question
    question = clean_sentence(question, stopwords=False)

    question_embedding = bert_model.encode([question])

    retrieveAndPrintFAQAnswer(question_embedding, sent_bertphrase_embeddings, df, cleaned_sentences, min_similarity)
