import pandas as pd
import os
import time
import pickle
import argparse
from openai import OpenAI
from tqdm import tqdm


MODEL = "text-embedding-3-large" #"text-embedding-ada-002"


def get_openai_embeddings(client, sentences, model="text-embedding-ada-002", batch_size=2048):
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding batches"):
        batch = sentences[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
        time.sleep(1)  # avoid rate limits
    return embeddings

def compute_and_save_embeddings_pickle(input_csv, output_path, openai_api_key):
    #openai.api_key = openai_api_key
    client = OpenAI(api_key=openai_api_key)

    # Read CSV and get sentences
    df = pd.read_csv(input_csv, sep='\t')
    all_sentences = df.iloc[:, 0].astype(str).tolist()
    all_words = df.iloc[:, 1].astype(str).tolist()

    for batch in range(1):
        batch = 'expanded'

        sentences = all_sentences#[batch::2]
        words = all_words#[batch::2]

        # Compute embeddings
        embeddings = get_openai_embeddings(client, sentences)

        # Combine sentences with their embeddings
        sentence_embedding_pairs = list(zip(sentences, words, embeddings))

        # Save as pickle
        with open(output_path + f".{batch}", 'wb') as f:
            pickle.dump(sentence_embedding_pairs, f)

        print(f"Saved {len(sentence_embedding_pairs)} sentence-embedding pairs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_path", help="Output path (should end in .pkl)")
    parser.add_argument("--openai_api_key", required=True, help="Your OpenAI API key")
    args = parser.parse_args()

    compute_and_save_embeddings_pickle(args.input_csv, args.output_path, args.openai_api_key)