import os
from typing import Dict, Iterator, List, Literal, Optional

from voyageai import Client

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

# Voyage AI by MongoDB embedding models served by the hosted API, see
# https://docs.voyageai.com/docs/embeddings. Per model:
#   dims:       output dimensions the API accepts (the first one is the native default)
#   max_tokens: tokens allowed across all inputs of one request, or None when Voyage
#               documents no such limit for the model
VOYAGE_MODELS: Dict[str, Dict] = {
    "voyage-4-large": {"dims": (1024, 256, 512, 2048), "max_tokens": 120_000},
    "voyage-4": {"dims": (1024, 256, 512, 2048), "max_tokens": 320_000},
    "voyage-4-lite": {"dims": (1024, 256, 512, 2048), "max_tokens": 1_000_000},
    "voyage-code-4": {"dims": (1024, 256, 512, 2048), "max_tokens": None},
    "voyage-3-large": {"dims": (1024, 256, 512, 2048), "max_tokens": 120_000},
    "voyage-3.5": {"dims": (1024, 256, 512, 2048), "max_tokens": 320_000},
    "voyage-3.5-lite": {"dims": (1024, 256, 512, 2048), "max_tokens": 1_000_000},
    "voyage-code-3": {"dims": (1024, 256, 512, 2048), "max_tokens": 120_000},
    "voyage-finance-2": {"dims": (1024,), "max_tokens": 120_000},
    "voyage-law-2": {"dims": (1024,), "max_tokens": 120_000},
}

# Contextualized embedding models (`/v1/contextualizedembeddings`), see
# https://docs.voyageai.com/docs/contextualized-chunk-embeddings
VOYAGE_CONTEXTUALIZED_MODELS: Dict[str, Dict] = {
    "voyage-context-4": {"dims": (1024, 256, 512, 2048), "max_tokens": 120_000},
    "voyage-context-3": {"dims": (1024, 256, 512, 2048), "max_tokens": 120_000},
}

DEFAULT_MODEL = "voyage-3.5"

# Voyage accepts at most 1000 inputs in a single request, regardless of model.
MAX_INPUTS_PER_REQUEST = 1000

# Maximum tokens per chunk (the context window of the contextualized models).
CONTEXTUALIZED_CHUNK_SIZE = 32_000


class VoyageEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or DEFAULT_MODEL
        self._contextualized = self.config.model in VOYAGE_CONTEXTUALIZED_MODELS
        if self.config.model in VOYAGE_MODELS:
            spec = VOYAGE_MODELS[self.config.model]
        elif self._contextualized:
            spec = VOYAGE_CONTEXTUALIZED_MODELS[self.config.model]
        else:
            supported = ", ".join(list(VOYAGE_MODELS) + list(VOYAGE_CONTEXTUALIZED_MODELS))
            raise ValueError(
                f"Unsupported Voyage AI by MongoDB model '{self.config.model}'. Supported models: {supported}"
            )

        self._max_tokens = spec["max_tokens"]
        # `output_dimension` is only sent when `embedding_dims` is set explicitly.
        self._output_dimension = self.config.embedding_dims
        if self._output_dimension is not None and self._output_dimension not in spec["dims"]:
            raise ValueError(
                f"embedding_dims={self._output_dimension} is not supported by '{self.config.model}'. "
                f"Supported dimensions: {sorted(spec['dims'])}"
            )
        self.config.embedding_dims = self._output_dimension or spec["dims"][0]

        # The official SDK routes keys with an "al-" prefix to ai.mongodb.com and all other keys
        # to api.voyageai.com, so no base URL is configured here.
        api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY")
        self.client = Client(api_key=api_key)

    @staticmethod
    def _input_type(memory_action: Optional[str]) -> str:
        """Voyage ``input_type`` for a Mem0 action: search is a query, add/update are documents."""
        return "query" if memory_action == "search" else "document"

    def _batches(self, texts: List[str]) -> Iterator[List[str]]:
        """Split ``texts`` into consecutive batches within Voyage's per-request limits.

        A batch holds at most 1000 inputs and, when Voyage documents a token limit for the
        model, at most that many tokens (counted with the SDK tokenizer). A single text above
        the token limit is sent on its own. Order is preserved.
        """
        if len(texts) == 1 or self._max_tokens is None:
            token_counts = None
        else:
            token_counts = [len(encoding.ids) for encoding in self.client.tokenize(texts, self.config.model)]

        batch: List[str] = []
        batch_tokens = 0
        for i, text in enumerate(texts):
            n_tokens = token_counts[i] if token_counts is not None else 0
            if batch and (
                len(batch) >= MAX_INPUTS_PER_REQUEST
                or (token_counts is not None and batch_tokens + n_tokens > self._max_tokens)
            ):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n_tokens
        if batch:
            yield batch

    def _embed_request(self, batch: List[str], input_type: str) -> List[List[float]]:
        if self._contextualized:
            return self._embed_contextualized(batch, input_type)
        kwargs = {"model": self.config.model, "input_type": input_type}
        if self._output_dimension is not None:
            kwargs["output_dimension"] = self._output_dimension
        return list(self.client.embed(batch, **kwargs).embeddings)

    def _embed_contextualized(self, batch: List[str], input_type: str) -> List[List[float]]:
        """Embed each string as its own document with the contextualized endpoint.

        Documents are sent as a flat ``list[str]`` with server-side auto-chunking (only the
        ``document`` input type supports it). Queries are sent without auto-chunking.
        """
        kwargs = {"inputs": batch, "model": self.config.model, "input_type": input_type}
        if input_type == "document":
            kwargs["enable_auto_chunking"] = True
            kwargs["chunk_size"] = CONTEXTUALIZED_CHUNK_SIZE
        if self._output_dimension is not None:
            kwargs["output_dimension"] = self._output_dimension
        response = self.client.contextualized_embed(**kwargs)

        embeddings = []
        for result in response.results:
            if len(result.embeddings) != 1:
                raise ValueError(
                    f"Voyage AI by MongoDB split a text into {len(result.embeddings)} chunks using model "
                    f"'{self.config.model}'; Mem0 stores one vector per text, so shorten the text to at most "
                    f"{CONTEXTUALIZED_CHUNK_SIZE} tokens"
                )
            embeddings.append(result.embeddings[0])
        return embeddings

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Voyage AI by MongoDB.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update".
                "search" embeds the text as a query, anything else as a document. Defaults to None.
        Returns:
            list: The embedding vector.
        """
        return self._embed_request([text], self._input_type(memory_action))[0]

    def embed_batch(self, texts, memory_action="add"):
        if not texts:
            return []
        input_type = self._input_type(memory_action)
        embeddings: List[List[float]] = []
        for batch in self._batches(texts):
            embeddings.extend(self._embed_request(batch, input_type))
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Voyage AI by MongoDB embed_batch() returned {len(embeddings)} embeddings for {len(texts)} texts"
                f" using model '{self.config.model}'"
            )
        return embeddings
