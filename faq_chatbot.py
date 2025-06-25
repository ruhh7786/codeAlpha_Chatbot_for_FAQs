import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the FAQ dataset
faqs = pd.read_csv('faqs.csv')

# Vectorize the questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faqs['Question'])

def get_best_answer(user_question):
    user_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vec, X)
    best_idx = similarities.argmax()
    best_score = similarities[0, best_idx]
    if best_score < 0.2:
        return "Sorry, I couldn't find a relevant answer."
    return faqs['Answer'].iloc[best_idx]

def main():
    print("Welcome to the FAQ Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        answer = get_best_answer(user_input)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    main() 