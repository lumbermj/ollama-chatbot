"""
RAG Chatbot using MongoDB and Ollama
Provides an interactive chat interface that retrieves relevant context
from MongoDB and streams responses from an Ollama language model.
"""

import asyncio
import sys

import mongodb  # Module for MongoDB retrieval functions
import ollama   # Ollama client for language model interaction

from src import config  # Configuration, including LANGUAGE_MODEL setting
from src.models import response_model


def build_instruction_prompt(chunks: list[str]) -> str:
    """Construct the system prompt using retrieved context."""
    # Format each context chunk for clarity
    context_text = "\n".join(f" - {chunk}" for chunk in chunks)
    return (
        "You are a helpful customer support chatbot.\n"
        "Use only the following pieces of context to answer the customer's question. "
        "The context contains previous customer queries and support responses.\n"
        "Don't make up any new information - only use what's provided:\n\n"
        f"{context_text}\n\n"
        "Provide a helpful, professional response based on this context."
    )


def display_retrieved(knowledge: list[response_model.RetrievedChunk]) -> None:
    """Prints each retrieved document with its similarity score and snippet."""
    print("Retrieved knowledge:")
    for item in knowledge:
        sim = item.get('similarity', 0)
        text = item.get('text', '')
        # Show only first 100 characters of text for preview
        snippet = text[:100].replace("\n", " ")
        print(f" - (similarity: {sim:.2f}) {snippet}...")


async def retrieve_knowledge(query: str, top_n: int = 3) -> list[response_model.RetrievedChunk]:
    """Fetches top_n relevant documents from MongoDB for the given query."""
    print("\nRetrieving relevant information from MongoDB...")
    results = await mongodb.retrieve_from_mongodb(query, top_n=top_n)
    if not results:
        print("Sorry, I couldn't retrieve relevant information.")
    return results


async def generate_response(query: str, context_chunks: list[str]) -> None:
    """Streams a response from the Ollama language model based on context."""
    instruction = build_instruction_prompt(context_chunks)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": query},
    ]
    print("\nChatbot response:")
    try:
        # Stream the response from the model
        stream = ollama.chat(
            model=config.LANGUAGE_MODEL,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            # Print each piece of the streamed response immediately
            print(chunk['message']['content'], end='', flush=True)
        print()  # Newline after response completes
    except Exception as e:
        print(f"Error generating response: {e}")


async def chat_with_rag() -> None:
    """Main loop: interactively accept user queries and respond using RAG."""
    print("\n=== MongoDB RAG Chatbot Ready ===")
    print("Type 'quit' to exit")

    while True:
        user_input = input("\nAsk me a question: ").strip()

        # Handle exit commands
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            print("Please enter a question.")
            continue

        # Retrieve context and display
        docs = await retrieve_knowledge(user_input)
        if not docs:
            continue
        display_retrieved(docs)

        # Extract text chunks for response generation
        chunks = [item['text'] for item in docs]

        # Generate and display chatbot response
        await generate_response(user_input, chunks)


if __name__ == "__main__":
    try:
        asyncio.run(chat_with_rag())
    except KeyboardInterrupt:
        print("\nChat session ended by user.")
        sys.exit(0)
