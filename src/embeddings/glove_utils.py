import numpy as np


def load_glove_from_file(glove_path: str, embedding_dim: int = 300):
    """
    Load GloVe embeddings from a text file.

    Download GloVe files from: https://nlp.stanford.edu/projects/glove/

    Args:
        glove_path: Path to GloVe text file (e.g., 'glove.6B.300d.txt')
        embedding_dim: Dimension of embeddings

    Returns:
        Dictionary mapping words to embedding vectors
    """
    print(f"Loading GloVe embeddings from: {glove_path}")
    print("This may take a few minutes...")

    embeddings_dict = {}

    with open(glove_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"  Loaded {line_num:,} words...")

            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")

            if len(vector) == embedding_dim:
                embeddings_dict[word] = vector

    print(f"✅ GloVe loaded: {len(embeddings_dict):,} words, {embedding_dim} dimensions")
    return embeddings_dict


class SimpleGloVeWrapper:
    """Simple wrapper to mimic gensim's KeyedVectors interface."""

    def __init__(self, glove_dict: dict, embedding_dim: int):
        self.embeddings = glove_dict
        self.vector_size = embedding_dim

    def __getitem__(self, word: str):
        """Get embedding for a word."""
        word = word.lower()
        if word not in self.embeddings:
            raise KeyError(f"Word '{word}' not in vocabulary")
        return self.embeddings[word]

    def __contains__(self, word: str):
        """Check if word is in vocabulary."""
        return word.lower() in self.embeddings

    def __len__(self):
        """Number of words in vocabulary."""
        return len(self.embeddings)


def get_word_embedding(word: str, glove_model, default_dim: int = 300):
    """
    Get embedding for a single word.

    Args:
        word: Word to embed
        glove_model: Loaded GloVe model (dict or SimpleGloVeWrapper)
        default_dim: Dimension for unknown words

    Returns:
        Word embedding vector
    """
    try:
        if isinstance(glove_model, dict):
            return glove_model.get(word.lower(), np.zeros(default_dim))
        else:
            return glove_model[word.lower()]
    except KeyError:
        return np.zeros(default_dim)


def embed_text_with_glove(text: str, glove_model, max_length: int = 30, aggregation: str = "mean"):
    """
    Embed text using GloVe.

    Args:
        text: Input text
        glove_model: Loaded GloVe model (dict or SimpleGloVeWrapper)
        max_length: Maximum number of words
        aggregation: "mean", "max", or "sequence"

    Returns:
        Embedded text (embed_dim,) or sequence of embeddings (max_length, embed_dim)
    """
    # Get embedding dimension
    if isinstance(glove_model, dict):
        embed_dim = len(next(iter(glove_model.values())))
    else:
        embed_dim = glove_model.vector_size

    # Tokenize (simple whitespace split)
    words = text.lower().split()[:max_length]

    # Get embeddings
    embeddings = []
    for word in words:
        emb = get_word_embedding(word, glove_model, embed_dim)
        embeddings.append(emb)

    if not embeddings:
        return np.zeros(embed_dim)

    embeddings = np.array(embeddings)  # (num_words, embed_dim)

    # Aggregate
    if aggregation == "mean":
        return embeddings.mean(axis=0)
    elif aggregation == "max":
        return embeddings.max(axis=0)
    elif aggregation == "sequence":
        # Pad to max_length
        if len(embeddings) < max_length:
            padding = np.zeros((max_length - len(embeddings), embed_dim))
            embeddings = np.vstack([embeddings, padding])
        return embeddings
    else:
        return embeddings.mean(axis=0)


def create_embedding_matrix(word_index: dict, glove_dict: dict, embedding_dim: int = 300):
    """
    Create an embedding matrix for a vocabulary.

    Args:
        word_index: Dictionary mapping words to indices
        glove_dict: GloVe embeddings dictionary
        embedding_dim: Dimension of embeddings

    Returns:
        Embedding matrix (vocab_size, embedding_dim)
    """
    vocab_size = len(word_index) + 1  # +1 for padding
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    found = 0
    for word, idx in word_index.items():
        embedding_vector = glove_dict.get(word.lower())
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
            found += 1

    print(
        f"Found embeddings for {found}/{len(word_index)} words ({100*found/len(word_index):.1f}%)"
    )
    return embedding_matrix
