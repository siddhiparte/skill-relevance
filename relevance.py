import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from functools import lru_cache
import warnings
from typing import List, Set, Union, Dict
from tqdm import tqdm
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SkillMatcher:
    def __init__(self, csv_path: str):
        """
        Initialize the SkillMatcher with optimized parameters
        """
        self.embedding_dim = 100
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.df: pd.DataFrame = None
        self.model = None
        self.classifier = None
        self.unique_skills: Set[str] = set()
        self._load_and_prepare_data(csv_path)

    @staticmethod
    @lru_cache(maxsize=10000)
    def _preprocess_text(text: str) -> str:
        """Cached text preprocessing"""
        return re.sub(r'[^\w\s.#+]', ' ', str(text).lower()).strip()

    def _load_and_prepare_data(self, csv_path: str) -> None:
        """Optimized data loading and preparation"""
        # Load data efficiently
        dtype_dict = {'skill_1': 'string', 'skill_2': 'string', 'label': 'int8'}
        self.df = pd.read_csv(csv_path, dtype=dtype_dict)

        # Vectorized preprocessing
        self.df['skill_1'] = self.df['skill_1'].apply(self._preprocess_text)
        self.df['skill_2'] = self.df['skill_2'].apply(self._preprocess_text)

        # Get unique skills efficiently
        self.unique_skills = set(pd.concat([
            self.df['skill_1'],
            self.df['skill_2']
        ]).unique())

        # Prepare FastText training data
        tokenized_skills = [
            word_tokenize(skill)
            for skill in self.unique_skills
            if isinstance(skill, str)
        ]

        # Train optimized FastText model
        self.model = FastText(
            sentences=tokenized_skills,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,
            workers=1,
            sg=1,
            epochs=5,
            alpha=0.025,
            min_alpha=0.0001,
            negative=5
        )

        # Efficient embedding precomputation
        self._precompute_all_embeddings()

    def _precompute_all_embeddings(self) -> None:
        """Efficient batch computation of embeddings"""
        # Process all skills at once
        embeddings = {}
        for skill in tqdm(self.unique_skills, desc="Computing embeddings"):
            if isinstance(skill, str):
                words = word_tokenize(skill)
                word_vectors = [
                    self.model.wv[word]
                    for word in words
                    if word in self.model.wv
                ]
                if word_vectors:
                    embeddings[skill] = np.mean(word_vectors, axis=0)
                else:
                    embeddings[skill] = np.zeros(self.embedding_dim)

        self.embedding_cache = embeddings
        self.df['skill_1_embedding'] = self.df['skill_1'].map(self.embedding_cache)

    @lru_cache(maxsize=10000)
    def _get_skill_embedding(self, skill: str) -> np.ndarray:
        """Optimized embedding retrieval with caching"""
        if skill in self.embedding_cache:
            return self.embedding_cache[skill]

        words = word_tokenize(skill)
        vectors = [
            self.model.wv[word]
            for word in words
            if word in self.model.wv
        ]
        embedding = (
            np.mean(vectors, axis=0)
            if vectors
            else np.zeros(self.embedding_dim)
        )
        self.embedding_cache[skill] = embedding
        return embedding

    def _create_feature_vectors_batch(
        self,
        skill_1_embeddings: np.ndarray,
        skill_2_embeddings: np.ndarray
    ) -> np.ndarray:
        """Vectorized feature creation for batches"""
        # Compute cosine similarity for the whole batch at once
        cos_sims = np.diagonal(
            cosine_similarity(skill_1_embeddings, skill_2_embeddings)
        )

        # Create feature vectors efficiently
        return np.column_stack((
            skill_1_embeddings,
            skill_2_embeddings,
            cos_sims
        ))

    def train_classifier(self, batch_size: int = 1000) -> float:
        """Optimized classifier training with batch processing"""
        total_samples = len(self.df)
        num_batches = (total_samples + batch_size - 1) // batch_size
        X_list = []

        for i in tqdm(range(num_batches), desc="Preparing features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            batch_df = self.df.iloc[start_idx:end_idx]

            # Get embeddings for the batch
            skill_1_embeddings = np.stack(batch_df['skill_1_embedding'].values)
            skill_2_embeddings = np.stack(
                batch_df['skill_2'].apply(self._get_skill_embedding).values
            )

            # Create feature vectors for the batch
            X_batch = self._create_feature_vectors_batch(
                skill_1_embeddings,
                skill_2_embeddings
            )
            X_list.append(X_batch)

        # Combine all batches
        X = np.concatenate(X_list)
        y = self.df['label'].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train classifier with optimal parameters
        self.classifier = LinearSVC(
            dual=False,
            C=1.0,
            max_iter=2000,
            class_weight='balanced'
        )
        self.classifier.fit(X_train, y_train)

        return accuracy_score(y_test, self.classifier.predict(X_test))

    def get_main_skills(
        self,
        skill_variations: Union[str, List[str]],
        batch_size: int = 100,
        threshold: float = 0.5
    ) -> Set[str]:
        """Optimized skill matching with batch processing"""
        if isinstance(skill_variations, str):
            skill_variations = [skill_variations]

        # Preprocess variations efficiently
        processed_variations = [
            self._preprocess_text(skill)
            for skill in skill_variations
        ]

        main_skills = set()
        num_batches = (len(processed_variations) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(processed_variations))
            batch = processed_variations[start_idx:end_idx]

            # Get embeddings for the batch
            variation_embeddings = np.stack([
                self._get_skill_embedding(var)
                for var in batch
            ])

            # Process each variation in the batch
            for idx, variation_embedding in enumerate(variation_embeddings):
                # Create features for all possible matches efficiently
                features = self._create_feature_vectors_batch(
                    np.stack([variation_embedding] * len(self.df)),
                    np.stack(self.df['skill_1_embedding'].values)
                )

                # Get predictions and confidence scores
                predictions = self.classifier.decision_function(features)
                max_confidence_idx = np.argmax(predictions)

                if predictions[max_confidence_idx] > threshold:
                    main_skills.add(self.df.iloc[max_confidence_idx]['skill_1'])

        return main_skills

# Example usage with performance monitoring
if __name__ == "__main__":
    import time

    # Initialize and train
    start_time = time.time()
    matcher = SkillMatcher('programming_languages.csv')
    print(f"Initialization time: {time.time() - start_time:.2f} seconds")

    # Train and measure accuracy
    start_time = time.time()
    accuracy = matcher.train_classifier()
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    print(f"Model Accuracy: {accuracy:.2f}")

    # Test with variations
    test_variations = ["c", "scala lang","python program","python","c++","c program"]
    start_time = time.time()
    main_skills = matcher.get_main_skills(test_variations)
    print(f"Prediction time: {time.time() - start_time:.2f} seconds")
    print("Main Skills:", main_skills)