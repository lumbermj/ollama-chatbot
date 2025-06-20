from datasets import load_dataset
import mongodb
import embeddings
from src import config


def load_and_process_dataset():
    """Load dataset and store in MongoDB."""
    # DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    # DATASET_SPLIT = "train"

    print("Loading Bitext customer support dataset from Hugging Face...")
    dataset = load_dataset(config.HF_DATASET_NAME, split=config.HF_DATASET_SPLIT)
    print(f'Loaded {len(dataset)} entries from {config.HF_DATASET_NAME}')

    successful_embeddings = 0

    for i, item in enumerate(dataset):
        text_field = item.get('text') or item.get('instruction') or item.get('input') or item.get('query') or item.get('question')
        response_field = item.get('response') or item.get('output') or item.get('answer') or item.get('target')

        if text_field and response_field:
            chunk_id = f"qa_{i}"
            qa_text = f"Q: {text_field}\nA: {response_field}"

            intent = item.get('intent', None)
            category = item.get('category', None)

            if embeddings.create_embedding(qa_text):
                successful_embeddings += 1
                mongodb.add_chunk_to_database(chunk_id, qa_text, intent, category, "qa")

        # Progress update
        if (i + 1) % 100 == 0:
            print(f'Processed {i + 1}/{len(dataset)} items ({successful_embeddings} successful)')

    print(f"Processing complete! Successfully stored {successful_embeddings} chunks in MongoDB")