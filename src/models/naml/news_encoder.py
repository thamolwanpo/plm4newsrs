# src/models/naml/news_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from src.embeddings import load_glove_from_file, SimpleGloVeWrapper, embed_text_with_glove
from src.models.components import BaseNewsEncoder, CNNEncoder, AdditiveAttention, CategoryEncoder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_weights(m: nn.Module):
    """Initialize weights following reference implementation."""
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class BERTTextEncoder(nn.Module):
    """
    TextEncoder for BERT (matches reference exactly).
    Uses [CLS] token → pooler → CNN → additive attention.
    """
    def __init__(self, word_embedding_dim, num_filters, window_size,
                 query_vector_dim, dropout_probability, config):
        super(BERTTextEncoder, self).__init__()
        self.config = config
        self.dropout_probability = dropout_probability

        # Load BERT
        bert = AutoModel.from_pretrained(config.model_name)
        self.dim = bert.config.hidden_size
        self.bert = bert

        # Freeze or fine-tune language model (matching original implementation)
        if not self.config.fine_tune_lm:
            print("Freezing language model parameters")
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            # Selective layer freezing: freeze all layers except the last N layers
            if hasattr(self.config, 'finetune_layers') and self.config.finetune_layers > 0:
                num_layers = len(self.bert.encoder.layer)
                print(f"Fine-tuning last {self.config.finetune_layers} layers out of {num_layers}")

                for i, layer in enumerate(self.bert.encoder.layer):
                    if i < num_layers - self.config.finetune_layers:
                        # Freeze this layer
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        # Fine-tune this layer
                        for param in layer.parameters():
                            param.requires_grad = True
            else:
                # Fine-tune all layers
                print("Fine-tuning all language model layers")
                for param in self.bert.parameters():
                    param.requires_grad = True

        # Pooler layer - projects to maintain_word_embedding_dim
        self.pooler = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(0.1),
            nn.LayerNorm(self.dim),
            nn.SiLU(),
        )
        self.pooler.apply(init_weights)

        # CNN on [CLS] token
        self.CNN = nn.Conv2d(1, num_filters, (window_size, word_embedding_dim),
                             padding=(int((window_size - 1) / 2), 0))

        # Additive attention (input_dim, hidden_dim)
        self.additive_attention = AdditiveAttention(num_filters, query_vector_dim)

    def forward(self, text):
        """
        Args:
            text: (batch, 2, seq_len) - stacked [input_ids, attention_mask]
                  OR (batch, seq_len) if only input_ids provided
        Returns:
            text_vector: (batch, num_filters)
        """
        # Handle different input formats
        if text.dim() == 3:
            # Format: (batch, 2, seq_len)
            news_input = {
                "input_ids": text[:, 0].long().to(device),
                "attention_mask": text[:, 1].long().to(device)
            }
        elif text.dim() == 2:
            # Format: (batch, seq_len) - only input_ids
            news_input = {
                "input_ids": text.long().to(device),
                "attention_mask": (text != 0).long().to(device)  # Auto-generate mask
            }
        else:
            raise ValueError(f"Expected 2D or 3D input, got {text.dim()}D tensor")

        # Get all token embeddings
        text_vector = self.bert(**news_input)[0]  # (batch, seq_len, hidden_dim)

        # Extract [CLS] token
        text_vector = text_vector[:, 0]  # (batch, hidden_dim)

        # Apply pooler
        text_vector = self.pooler(text_vector)  # (batch, hidden_dim)

        # Reshape for CNN and apply
        convoluted_text_vector = self.CNN(
            text_vector.unsqueeze(1).unsqueeze(2).float()
        ).squeeze(dim=3)  # (batch, num_filters, 1)

        # ReLU + dropout
        activated_text_vector = F.dropout(
            F.relu(convoluted_text_vector),
            p=self.dropout_probability,
            training=self.training
        )

        # Additive attention
        text_vector, _ = self.additive_attention(
            activated_text_vector.transpose(1, 2)
        )  # (batch, num_filters)

        return text_vector


class GloVeTextEncoder(nn.Module):
    """
    TextEncoder for GloVe (matches reference exactly).
    Uses word embeddings → linear → CNN → additive attention.
    """
    def __init__(self, word_embedding_dim, num_filters, window_size,
                 query_vector_dim, dropout_probability, linear_dim, maintain_word_embedding_dim):
        super(GloVeTextEncoder, self).__init__()
        self.dropout_probability = dropout_probability

        # Projection layer
        self.linear = nn.Linear(word_embedding_dim, linear_dim)

        # CNN
        self.CNN = nn.Conv2d(1, num_filters, (window_size, word_embedding_dim),
                             padding=(int((window_size - 1) / 2), 0))

        # Additive attention (input_dim, hidden_dim)
        self.additive_attention = AdditiveAttention(num_filters, query_vector_dim)

    def forward(self, text):
        """
        Args:
            text: (batch, num_words, word_embedding_dim)
        Returns:
            text_vector: (batch, num_filters)
        """
        # Linear projection
        projected_text_embed = self.linear(text.float())
        text_vector = F.dropout(
            projected_text_embed,
            p=self.dropout_probability,
            training=self.training
        )

        # CNN
        convoluted_text_vector = self.CNN(
            text_vector.unsqueeze(dim=1)
        ).squeeze(dim=3)  # (batch, num_filters, num_words)

        # ReLU + dropout
        activated_text_vector = F.dropout(
            F.relu(convoluted_text_vector),
            p=self.dropout_probability,
            training=self.training
        )

        # Additive attention
        text_vector, _ = self.additive_attention(
            activated_text_vector.transpose(1, 2)
        )  # (batch, num_filters)

        return text_vector


class ElementEncoder(nn.Module):
    """Element encoder for category/subcategory (matches reference)."""
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super(ElementEncoder, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class NAMLNewsEncoder(BaseNewsEncoder):
    """
    NAML News Encoder with Attentive Multi-View Learning.

    Matches reference implementation with support for both BERT and GloVe.

    Architecture:
    - BERT: [CLS] token → pooler → CNN → additive attention
    - GloVe: embeddings → linear → CNN → additive attention
    - Category/Subcategory: embedding → linear
    - Final: multi-view attention aggregation

    Based on: "Neural News Recommendation with Attentive Multi-View Learning"
    (Wu et al., IJCAI 2019)
    """

    def _init_glove_encoder(self):
        """Initialize GloVe-based encoder."""
        print(f"Using GloVe embeddings from: {self.config.glove_file_path}")

        # Load GloVe from file
        glove_dict = load_glove_from_file(self.config.glove_file_path, self.config.glove_dim)
        self.glove_model = SimpleGloVeWrapper(glove_dict, self.config.glove_dim)

        # Linear projection layer (to match original GloVe implementation)
        if hasattr(self.config, 'linear_dim'):
            self.title_linear = nn.Linear(self.config.glove_dim, self.config.linear_dim)
            self.hidden_size = self.config.linear_dim
        else:
            # Fallback: no projection
            self.title_linear = None
            self.hidden_size = self.config.glove_dim

    def __init__(self, config):
        super(NAMLNewsEncoder, self).__init__(config)
        self.config = config

        # Determine which views are enabled
        self.use_title = config.use_title
        self.use_body = config.use_body
        self.use_category = config.use_category
        self.use_subcategory = config.use_subcategory

        # Count active views
        num_views = sum([self.use_title, self.use_body, self.use_category, self.use_subcategory])
        if num_views == 0:
            raise ValueError("At least one view must be enabled")

        # Determine if using BERT or GloVe
        self.use_bert = "glove" not in config.model_name.lower()

        if not self.use_bert:
            self._init_glove_encoder()

        # Create text encoders for enabled views
        self.text_encoders = nn.ModuleDict()

        if self.use_bert:
            # BERT text encoders
            if self.use_title:
                self.text_encoders['title'] = BERTTextEncoder(
                    word_embedding_dim=config.word_embedding_dim,
                    num_filters=config.num_filters,
                    window_size=config.window_sizes[0] if isinstance(config.window_sizes, (list, tuple)) else 3,
                    query_vector_dim=config.query_vector_dim,
                    dropout_probability=config.drop_rate,
                    config=config
                )
            if self.use_body:
                self.text_encoders['text'] = BERTTextEncoder(
                    word_embedding_dim=config.word_embedding_dim,
                    num_filters=config.num_filters,
                    window_size=config.window_sizes[0] if isinstance(config.window_sizes, (list, tuple)) else 3,
                    query_vector_dim=config.query_vector_dim,
                    dropout_probability=config.drop_rate,
                    config=config
                )
        else:
            # GloVe text encoders
            if self.use_title:
                self.text_encoders['title'] = GloVeTextEncoder(
                    word_embedding_dim=config.word_embedding_dim,
                    num_filters=config.num_filters,
                    window_size=config.window_sizes[0] if isinstance(config.window_sizes, (list, tuple)) else 3,
                    query_vector_dim=config.query_vector_dim,
                    dropout_probability=config.drop_rate,
                    linear_dim=config.word_embedding_dim,
                    maintain_word_embedding_dim=config.word_embedding_dim
                )
            if self.use_body:
                self.text_encoders['text'] = GloVeTextEncoder(
                    word_embedding_dim=config.word_embedding_dim,
                    num_filters=config.num_filters,
                    window_size=config.window_sizes[0] if isinstance(config.window_sizes, (list, tuple)) else 3,
                    query_vector_dim=config.query_vector_dim,
                    dropout_probability=config.drop_rate,
                    linear_dim=config.word_embedding_dim,
                    maintain_word_embedding_dim=config.word_embedding_dim
                )

        # Category embedding (shared)
        category_embedding = nn.Embedding(
            config.num_categories,
            config.category_embedding_dim,
            padding_idx=0
        )

        # Create element encoders for enabled views
        self.element_encoders = nn.ModuleDict()

        if self.use_category:
            self.element_encoders['category'] = ElementEncoder(
                embedding=category_embedding,
                linear_input_dim=config.category_embedding_dim,
                linear_output_dim=config.num_filters
            )

        if self.use_subcategory:
            self.element_encoders['subcategory'] = ElementEncoder(
                embedding=category_embedding,
                linear_input_dim=config.category_embedding_dim,
                linear_output_dim=config.num_filters
            )

        # Final attention (if multiple views)
        if num_views > 1:
            self.final_attention = AdditiveAttention(
                config.num_filters,
                config.query_vector_dim
            )
        else:
            self.final_attention = None

        self.output_size = config.num_filters

        # For debugging: store view representations
        self.store_view_representations = False
        self.last_view_representations = {}

    def forward(
        self,
        title_input_ids=None,
        title_attention_mask=None,
        title_text_list=None,
        body_input_ids=None,
        body_attention_mask=None,
        body_text_list=None,
        category_ids=None,
        subcategory_ids=None,
        device_indicator=None,
    ):
        """
        Forward pass - handles both GloVe and transformer modes.

        Args (Transformer mode):
            title_input_ids: (batch, 2, seq_len) - stacked [input_ids, attention_mask]
            title_attention_mask: Not used (kept for compatibility)
            body_input_ids: (batch, 2, seq_len) - stacked [input_ids, attention_mask] (optional)
            body_attention_mask: Not used (kept for compatibility)
            category_ids: (batch,) - optional
            subcategory_ids: (batch,) - optional

        Args (GloVe mode):
            title_text_list: (batch, num_words, word_embedding_dim)
            body_text_list: (batch, num_words, word_embedding_dim) - optional
            category_ids: (batch,) - optional
            subcategory_ids: (batch,) - optional
            device_indicator: Tensor on correct device

        Returns:
            final_news_vector: (batch, num_filters)
        """
        # Build news dict from kwargs (matching reference input format)
        news = {}

        # Handle title/text view
        if self.use_bert:
            if title_input_ids is not None:
                news['title'] = title_input_ids
            if body_input_ids is not None:
                news['text'] = body_input_ids
        else:
            if title_text_list is not None:
                # Assuming title_text_list is List[str] containing raw news texts
                if isinstance(title_text_list, list):
                    title_embeddings = []
                    for text in title_text_list:
                        # (max_seq_length, embed_dim)
                        emb = embed_text_with_glove(
                            text, self.glove_model, self.config.max_title_length, aggregation="sequence"
                        )
                        title_embeddings.append(torch.tensor(emb, dtype=torch.float32))
                    # Stack them into a single tensor (N, L, D)
                    news['title'] = torch.stack(title_embeddings)
                else:
                    # If it's already a tensor, use it directly
                    news['title'] = title_text_list
                    
            if body_text_list is not None:
                if isinstance(body_text_list, list):
                    body_embeddings = []
                    for text in body_text_list:
                        emb = embed_text_with_glove(
                            text, self.glove_model, self.config.max_body_length, aggregation="sequence"
                        )
                        body_embeddings.append(torch.tensor(emb, dtype=torch.float32))
                    # Stack them into a single tensor (N, L, D)
                    news['text'] = torch.stack(body_embeddings)
                else:
                    news['text'] = body_text_list

        # Handle category/subcategory
        if category_ids is not None:
            news['category'] = category_ids
        if subcategory_ids is not None:
            news['subcategory'] = subcategory_ids

        # Encode text views (title, text)
        text_vectors = []
        for name, encoder in self.text_encoders.items():
            if name in news:
                vec = encoder(news[name].to(device))
                text_vectors.append(vec)

                # Store for debugging
                if self.store_intermediate_outputs:
                    self._store_output(f"{name}_vector", vec)

        # Encode element views (category, subcategory)
        element_vectors = []
        for name, encoder in self.element_encoders.items():
            if name in news:
                vec = encoder(news[name].to(device))
                element_vectors.append(vec)

                # Store for debugging
                if self.store_intermediate_outputs:
                    self._store_output(f"{name}_vector", vec)

        # Combine all vectors
        all_vectors = text_vectors + element_vectors

        # Store individual view representations for analysis
        if self.store_view_representations:
            for i, (view_name, view_vec) in enumerate(
                zip(
                    list(self.text_encoders.keys()) + list(self.element_encoders.keys()),
                    all_vectors
                )
            ):
                self.last_view_representations[view_name] = view_vec.detach().cpu()

        # Aggregate views
        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
            view_attention_weights = None
        else:
            # Stack and apply final attention
            stacked = torch.stack(all_vectors, dim=1)  # (batch, num_views, num_filters)

            if self.store_intermediate_outputs:
                self._store_output("stacked_views", stacked)

            final_news_vector, view_attention_weights = self.final_attention(stacked)

            if self.store_intermediate_outputs:
                self._store_output("view_attention_weights", view_attention_weights)

        if self.store_intermediate_outputs:
            self._store_output("final_news_embedding", final_news_vector)

        return final_news_vector

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.output_size