"""
Module: Dataset Loader and Processor

This script loads a customer support dataset from Hugging Face,
creates embeddings for each Q/A pair, and stores them in MongoDB.
"""

# External libraries for dataset loading and MongoDB interaction
from datasets import load_dataset
import mongodb
import embeddings

# Configuration containing dataset identifiers and other settings
from src import config


def load_and_process_dataset():
    """
    Load a dataset from Hugging Face, generate embeddings for each question-answer pair,
    and store them in the MongoDB collection.
    """
    # Inform the user that the dataset loading process is starting
    print("Loading Bitext customer support dataset from Hugging Face...")

    # Load the specified split of the dataset using configuration constants
    dataset = load_dataset(
        config.HF_DATASET_NAME,
        split=config.HF_DATASET_SPLIT
    )
    print(f"Loaded {len(dataset)} entries from {config.HF_DATASET_NAME}")

    # Counter for successful embedding creations
    successful_embeddings = 0

    # Iterate over the dataset entries, extracting text and response fields
    for i, item in enumerate(dataset):
        # Determine the question text from possible field names
        text_field = (
            item.get('text') or
            item.get('instruction') or
            item.get('input') or
            item.get('query') or
            item.get('question')
        )
        # Determine the answer text from possible field names
        response_field = (
            item.get('response') or
            item.get('output') or
            item.get('answer') or
            item.get('target')
        )

        # Process only if both question and answer are present
        if text_field and response_field:
            # Construct a unique identifier for this QA pair
            chunk_id = f"qa_{i}"
            # Combine into a single text block for embedding
            qa_text = f"Q: {text_field}\nA: {response_field}"

            # Optional metadata fields
            intent = item.get('intent', None)
            category = item.get('category', None)

            # Create embedding for the QA text; if successful, add to MongoDB
            if embeddings.create_embedding(qa_text):
                successful_embeddings += 1
                mongodb.add_chunk_to_database(
                    chunk_id=chunk_id,
                    text=qa_text,
                    intent=intent,
                    category=category,
                    chunk_type="qa"
                )

        # Print progress every 100 items
        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(dataset)} items "
                f"({successful_embeddings} embeddings stored)"
            )

    # Final summary of processing results
    print(
        f"Processing complete! Successfully stored "
        f"{successful_embeddings} chunks in MongoDB"
    )
