import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
# import pyLDAvis.sklearn
import nltk
from nltk.corpus import stopwords
import warnings
nltk.download('stopwords')

warnings.filterwarnings('ignore')


class LDATopicModeling:
    def __init__(self, n_topics=10, max_features=2000):
        """
        Initialize LDA topic model

        Parameters:
        n_topics: number of topics
        max_features: maximum number of words to keep
        """
        # Download stopwords if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.n_topics = n_topics
        self.max_features = max_features
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = CountVectorizer(max_features=max_features,
                                          stop_words=self.stop_words)
        self.lda = LatentDirichletAllocation(n_components=n_topics,
                                             random_state=42,
                                             n_jobs=-1)

    def fit_transform(self, documents):
        """
        Train model and transform documents

        Parameters:
        documents: list of documents
        """
        # Document vectorization
        self.doc_term_matrix = self.vectorizer.fit_transform(documents)

        # Train LDA model
        self.doc_topic_dist = self.lda.fit_transform(self.doc_term_matrix)

        return self.doc_topic_dist

    def get_topic_words(self, n_words=10):
        """
        Get keywords for each topic

        Parameters:
        n_words: number of keywords to return per topic
        """
        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = []

        # Get word distribution for each topic
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words.append({
                'Topic': topic_idx,
                'Words': top_words
            })

        return pd.DataFrame(topic_words)

    def get_document_topics(self, threshold=0.1):
        """
        Get topic distribution for each document

        Parameters:
        threshold: probability threshold for topics
        """
        doc_topics = []

        for doc_idx, doc_dist in enumerate(self.doc_topic_dist):
            # Get topics with probability above threshold
            major_topics = [{'Topic': topic_idx, 'Probability': prob}
                            for topic_idx, prob in enumerate(doc_dist)
                            if prob > threshold]

            # Sort by probability in descending order
            major_topics = sorted(major_topics,
                                  key=lambda x: x['Probability'],
                                  reverse=True)

            doc_topics.append({
                'Document_ID': doc_idx,
                'Major_Topics': major_topics
            })

        return pd.DataFrame(doc_topics)

    def visualize_topics(self):
        """Generate interactive topic visualization"""
        return pyLDAvis.sklearn.prepare(self.lda,
                                        self.doc_term_matrix,
                                        self.vectorizer)


# Usage example
if __name__ == "__main__":
    import json
    documents = []
    input_filename = "../dataset/CORAL/deduplicate_passage_corpus.json"
    with open(input_filename, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                ref_string = data.get('ref_string', '')
                documents.append(ref_string)
            except:
                pass

    # Initialize model
    lda_model = LDATopicModeling(n_topics=5, max_features=1000)

    # Train model
    doc_topics = lda_model.fit_transform(documents)

    # Get topic words
    topic_words = lda_model.get_topic_words(n_words=10)
    print("\nTopic word distribution:")
    print(topic_words)

    # Get document topics
    doc_topics = lda_model.get_document_topics(threshold=0.1)
    print("\nDocument topic distribution:")
    print(doc_topics)

    # Generate visualization
    vis = lda_model.visualize_topics()
    # Save visualization
    pyLDAvis.save_html(vis, 'lda_visualization.html')