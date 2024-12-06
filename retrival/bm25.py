from rank_bm25 import BM25Okapi
from typing import List, Union
import numpy as np
from collections import Counter
import re
import json


class DocumentSearchBM25:
    """
    A document search class using BM25 algorithm for ranking documents based on relevance to a query.

    Attributes:
        documents (List[str]): List of documents to search through
        bm25 (BM25Okapi): BM25 index object
        tokenized_docs (List[List[str]]): Preprocessed and tokenized documents
    """

    def __init__(self, documents: List[str]):
        """
        Initialize the search engine with a list of documents.

        Args:
            documents (List[str]): List of documents to index
        """
        self.documents = documents
        self.tokenized_docs = [self._preprocess_text(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the text by converting to lowercase, removing special characters,
        and tokenizing into words.

        Args:
            text (str): Input text to preprocess

        Returns:
            List[str]: List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = text.split()

        return tokens

    def search(self, query: str, limit_num: int = 5) -> List[tuple]:
        """
        Search for documents most relevant to the query using BM25 ranking.

        Args:
            query (str): Search query
            limit_num (int): Maximum number of results to return

        Returns:
            List[tuple]: List of tuples containing (document_index, score, document_text)
                        sorted by relevance score in descending order
        """
        # Input validation
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if limit_num < 1:
            raise ValueError("limit_num must be positive")

        # Preprocess the query
        tokenized_query = self._preprocess_text(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top-k scores
        top_k_indices = np.argsort(scores)[::-1][:limit_num]

        # Create result list with (index, score, document) tuples
        results = [
            (idx, scores[idx], self.documents[idx])
            for idx in top_k_indices
            if scores[idx] > 0  # Only include documents with positive scores
        ]

        return results

    def get_document_statistics(self) -> dict:
        """
        Get basic statistics about the document collection.

        Returns:
            dict: Dictionary containing various statistics about the documents
        """
        # Calculate document lengths
        doc_lengths = [len(doc) for doc in self.tokenized_docs]

        # Calculate vocabulary statistics
        all_tokens = [token for doc in self.tokenized_docs for token in doc]
        vocabulary = Counter(all_tokens)

        return {
            'num_documents': len(self.documents),
            'avg_document_length': np.mean(doc_lengths),
            'vocabulary_size': len(vocabulary),
            'total_tokens': len(all_tokens)
        }


# Example usage
def demo_search():
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a popular programming language",
        "Machine learning algorithms help in data analysis",
        "The lazy cat sleeps all day",
        "Data science involves programming and statistics"
    ]

    # Initialize search engine
    search_engine = DocumentSearchBM25(documents)

    # Example search
    query = "programming language"
    limit_num = 3

    try:
        results = search_engine.search(query, limit_num)
        print(f"\nTop {limit_num} results for query: '{query}'")
        for idx, score, doc in results:
            print(f"Document {idx} (Score: {score:.4f}): {doc}")

        # Print statistics
        stats = search_engine.get_document_statistics()
        print("\nDocument Collection Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error during search: {str(e)}")

def bm25(question):
    documents = []
    file_path = r"../dataset/CORAL/deduplicate_passage_corpus.json"
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                ref_string = data.get('ref_string', '')
                documents.append(ref_string)
            except:
                pass

    # Initialize search engine
    search_engine = DocumentSearchBM25(documents)

    # Example search
    query = question
    limit_num = 10
    bm25_retrival_document = []
    try:
        results = search_engine.search(query, limit_num)
        print(f"\nTop {limit_num} results for query: '{query}'")
        for idx, score, doc in results:
            bm25_retrival_document.append(doc)
            # print(f"Document {idx} (Score: {score:.4f}): {doc}")

        # Print statistics
        # stats = search_engine.get_document_statistics()
        # print("\nDocument Collection Statistics:")
        # for key, value in stats.items():
        #     print(f"{key}: {value}")

    except Exception as e:
        print(f"Error during search: {str(e)}")
    return bm25_retrival_document

if __name__ == "__main__":
    question = "Can you tell me more about his history with the Albania U21 national team?"
    bm25(question)