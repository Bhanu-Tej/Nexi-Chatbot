
import nltk
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

# Sample corpus for Nexi
corpus = [
    "Hello! I'm Nexi, your friendly AI assistant. How can I help you today?",
    "My name is Nexi. I'm here to answer your questions.",
    "I can help you with basic queries about Artificial Intelligence, Machine Learning, or just have a friendly chat.",
    "Artificial Intelligence is the simulation of human intelligence in machines.",
    "Goodbye! Have a wonderful day!"
]

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Sentiment words
happy_words = ['happy', 'great', 'good', 'fantastic', 'awesome']
sad_words = ['sad', 'bad', 'unhappy', 'upset', 'angry']
jokes = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "I told my computer I needed a break, and it gave me a Kit-Kat!",
    "Why was the math book sad? Because it had too many problems!"
]

def chatbot_response(user_input):
    lower_input = user_input.lower()

    if any(word in lower_input for word in happy_words):
        return "I'm so glad to hear that! 😊"

    elif any(word in lower_input for word in sad_words):
        return "Oh no! I'm here for you. Here's a virtual hug 🤗"

    elif "joke" in lower_input:
        return random.choice(jokes)

    else:
        corpus.append(user_input)
        vectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
        X = vectorizer.fit_transform(corpus)
        vals = cosine_similarity(X[-1], X)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        score = flat[-2]
        corpus.pop()

        if score > 0.5:
            return corpus[idx]
        else:
            return "I'm still learning! Could you rephrase that for me?"

print("Nexi: Hello! I'm Nexi. Type 'bye' anytime to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("Nexi: Goodbye! Take care!")
        break
    else:
        print("Nexi:", chatbot_response(user_input))
