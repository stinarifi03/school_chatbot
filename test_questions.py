from app import EpokaChatbot

# Initialize chatbot
chatbot = EpokaChatbot()
chatbot.initialize(rebuild_index=False)

# Test questions
test_questions = [
    "When is the winter break?",
    "When are final exams?",
    "What are the attendance requirements?",
    "What is the tuition fee?",
    "When does registration start?"
]

for question in test_questions:
    print(f"\nQ: {question}")
    response = chatbot.query(question)
    print(f"A: {response['answer']}\n")
    print(f"Confidence: {response['citations'][0]['score'] if response['citations'] else 'N/A'}")
    print("-" * 80)