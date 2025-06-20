import mongodb
import ollama

from src import config


def chat_with_rag():
    """Interactive chat interface using MongoDB RAG."""
    print("\n=== MongoDB RAG Chatbot Ready ===")
    print("Type 'quit' to exit")

    while True:
        input_query = input('\nAsk me a question: ').strip()

        if input_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not input_query:
            print("Please enter a question.")
            continue

        print("Retrieving relevant information from MongoDB...")
        retrieved_knowledge = mongodb.retrieve_from_mongodb(input_query, top_n=3)

        if not retrieved_knowledge:
            print("Sorry, I couldn't retrieve relevant information.")
            continue

        print('Retrieved knowledge:')
        for item in retrieved_knowledge:
            print(f" - (similarity: {item['similarity']:.2f}) {item['text'][:100]}...")

        # Prepare context for the language model
        context_chunks = [item['text'] for item in retrieved_knowledge]

        instruction_prompt = f'''You are a helpful customer support chatbot.
Use only the following pieces of context to answer the customer's question. The context contains previous customer queries and support responses.
Don't make up any new information - only use what's provided:

{chr(10).join([f' - {chunk}' for chunk in context_chunks])}

Provide a helpful, professional response based on this context.'''

        try:
            stream = ollama.chat(
                model=config.LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': input_query},
                ],
                stream=True,
            )

            print('\nChatbot response:')
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print()  # New line after response

        except Exception as e:
            print(f"Error generating response: {e}")