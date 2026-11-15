from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.embeddings.voyageai import MAX_INPUTS_PER_REQUEST, VoyageEmbedding
from mem0.utils.factory import EmbedderFactory


@pytest.fixture
def mock_client():
    with patch("mem0.embeddings.voyageai.Client") as client_cls:
        client = Mock()
        client_cls.return_value = client
        yield client


def _encodings(token_counts):
    return [SimpleNamespace(ids=[0] * n) for n in token_counts]


def _plain_response(texts):
    # Encode each text's position in the vector so ordering mistakes are visible.
    return Mock(embeddings=[[float(t.split("-")[1])] for t in texts])


def _contextualized_response(n_documents, chunks_per_document=1):
    results = [SimpleNamespace(index=i, embeddings=[[float(i)]] * chunks_per_document) for i in range(n_documents)]
    return Mock(results=list(reversed(results)))  # out of order on purpose


def test_unknown_model_is_rejected(mock_client):
    with pytest.raises(ValueError, match="Unsupported Voyage AI model 'voyage-nope-1'"):
        VoyageEmbedding(BaseEmbedderConfig(model="voyage-nope-1"))


@pytest.mark.parametrize("model", ["voyage-finance-2", "voyage-law-2"])
def test_fixed_dimension_model_rejects_other_dims(mock_client, model):
    with pytest.raises(ValueError, match="not supported"):
        VoyageEmbedding(BaseEmbedderConfig(model=model, embedding_dims=256))


def test_unsupported_dims_for_matryoshka_model_are_rejected(mock_client):
    with pytest.raises(ValueError, match="not supported"):
        VoyageEmbedding(BaseEmbedderConfig(model="voyage-4", embedding_dims=768))


def test_output_dimension_sent_only_when_requested(mock_client):
    mock_client.embed.return_value = Mock(embeddings=[[0.1]])

    default = VoyageEmbedding(BaseEmbedderConfig(model="voyage-4"))
    assert default.config.embedding_dims == 1024
    default.embed("a")
    assert "output_dimension" not in mock_client.embed.call_args.kwargs

    explicit = VoyageEmbedding(BaseEmbedderConfig(model="voyage-4", embedding_dims=512))
    explicit.embed("a")
    assert mock_client.embed.call_args.kwargs["output_dimension"] == 512


@pytest.mark.parametrize(
    "action,expected", [("search", "query"), ("add", "document"), ("update", "document"), (None, "document")]
)
def test_input_type_mapping(mock_client, action, expected):
    mock_client.embed.return_value = Mock(embeddings=[[0.1]])
    VoyageEmbedding(BaseEmbedderConfig(model="voyage-4")).embed("a", memory_action=action)
    assert mock_client.embed.call_args.kwargs["input_type"] == expected


def test_single_embed_does_not_tokenize(mock_client):
    mock_client.embed.return_value = Mock(embeddings=[[0.1]])
    VoyageEmbedding(BaseEmbedderConfig(model="voyage-4")).embed_batch(["only one"])
    mock_client.tokenize.assert_not_called()


def test_batches_split_at_model_token_limit_and_keep_order(mock_client):
    # voyage-4-large: 120_000 tokens per request. 70k + 40k fit; the next 50k does not.
    texts = ["t-0", "t-1", "t-2", "t-3"]
    mock_client.tokenize.return_value = _encodings([70_000, 40_000, 50_000, 10_000])
    mock_client.embed.side_effect = lambda batch, **kw: _plain_response(batch)

    result = VoyageEmbedding(BaseEmbedderConfig(model="voyage-4-large")).embed_batch(texts)

    assert [call.args[0] for call in mock_client.embed.call_args_list] == [["t-0", "t-1"], ["t-2", "t-3"]]
    assert result == [[0.0], [1.0], [2.0], [3.0]]
    mock_client.tokenize.assert_called_once_with(texts, "voyage-4-large")


def test_token_limit_differs_per_model(mock_client):
    texts = ["t-0", "t-1"]
    mock_client.tokenize.return_value = _encodings([200_000, 200_000])
    mock_client.embed.side_effect = lambda batch, **kw: _plain_response(batch)

    VoyageEmbedding(BaseEmbedderConfig(model="voyage-4-large")).embed_batch(texts)  # 120K limit
    assert mock_client.embed.call_count == 2

    mock_client.embed.reset_mock()
    VoyageEmbedding(BaseEmbedderConfig(model="voyage-4-lite")).embed_batch(texts)  # 1M limit
    assert mock_client.embed.call_count == 1


def test_batches_split_at_1000_inputs(mock_client):
    texts = [f"t-{i}" for i in range(MAX_INPUTS_PER_REQUEST + 5)]
    mock_client.tokenize.return_value = _encodings([1] * len(texts))
    mock_client.embed.side_effect = lambda batch, **kw: _plain_response(batch)

    result = VoyageEmbedding(BaseEmbedderConfig(model="voyage-4")).embed_batch(texts)

    assert [len(call.args[0]) for call in mock_client.embed.call_args_list] == [MAX_INPUTS_PER_REQUEST, 5]
    assert result == [[float(i)] for i in range(len(texts))]


def test_oversized_single_input_is_sent_alone(mock_client):
    texts = ["t-0", "t-1", "t-2"]
    mock_client.tokenize.return_value = _encodings([10, 500_000, 10])
    mock_client.embed.side_effect = lambda batch, **kw: _plain_response(batch)

    VoyageEmbedding(BaseEmbedderConfig(model="voyage-4-large")).embed_batch(texts)

    assert [call.args[0] for call in mock_client.embed.call_args_list] == [["t-0"], ["t-1"], ["t-2"]]


def test_model_without_documented_token_limit_is_only_capped_by_input_count(mock_client):
    texts = [f"t-{i}" for i in range(3)]
    mock_client.embed.side_effect = lambda batch, **kw: _plain_response(batch)

    VoyageEmbedding(BaseEmbedderConfig(model="voyage-code-4")).embed_batch(texts)

    mock_client.tokenize.assert_not_called()
    assert mock_client.embed.call_count == 1


def test_contextualized_documents_use_flat_list_with_auto_chunking(mock_client):
    mock_client.contextualized_embed.side_effect = lambda **kw: _contextualized_response(len(kw["inputs"]))
    mock_client.tokenize.return_value = _encodings([5, 5])

    result = VoyageEmbedding(BaseEmbedderConfig(model="voyage-context-4", embedding_dims=512)).embed_batch(
        ["first", "second"]
    )

    mock_client.contextualized_embed.assert_called_once_with(
        inputs=["first", "second"],
        model="voyage-context-4",
        input_type="document",
        enable_auto_chunking=True,
        chunk_size=32_000,
        output_dimension=512,
    )
    assert result == [[0.0], [1.0]]  # re-sorted by result index


def test_contextualized_queries_do_not_use_auto_chunking(mock_client):
    mock_client.contextualized_embed.side_effect = lambda **kw: _contextualized_response(len(kw["inputs"]))

    VoyageEmbedding(BaseEmbedderConfig(model="voyage-context-3")).embed("what?", memory_action="search")

    kwargs = mock_client.contextualized_embed.call_args.kwargs
    assert kwargs["input_type"] == "query"
    assert kwargs["inputs"] == ["what?"]
    assert "enable_auto_chunking" not in kwargs
    assert "chunk_size" not in kwargs


def test_contextualized_text_split_into_several_chunks_raises(mock_client):
    mock_client.contextualized_embed.return_value = _contextualized_response(1, chunks_per_document=2)

    with pytest.raises(ValueError, match="split input 0 into 2 chunks"):
        VoyageEmbedding(BaseEmbedderConfig(model="voyage-context-4")).embed("very long text")


def test_embed_batch_count_mismatch_raises(mock_client):
    mock_client.tokenize.return_value = _encodings([1, 1])
    mock_client.embed.return_value = Mock(embeddings=[[0.1]])

    with pytest.raises(ValueError, match="returned 1 embeddings for 2 texts"):
        VoyageEmbedding(BaseEmbedderConfig(model="voyage-4")).embed_batch(["a", "b"])


def test_embed_batch_empty_makes_no_request(mock_client):
    assert VoyageEmbedding(BaseEmbedderConfig(model="voyage-4")).embed_batch([]) == []
    mock_client.embed.assert_not_called()


def test_api_key_comes_from_config_or_environment(monkeypatch):
    with patch("mem0.embeddings.voyageai.Client") as client_cls:
        monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
        VoyageEmbedding(BaseEmbedderConfig())
        client_cls.assert_called_with(api_key="env-key")
        VoyageEmbedding(BaseEmbedderConfig(api_key="config-key"))
        client_cls.assert_called_with(api_key="config-key")


def test_provider_is_registered_in_config_and_factory(mock_client):
    assert EmbedderConfig(provider="voyageai", config={}).provider == "voyageai"
    assert isinstance(EmbedderFactory.create("voyageai", {"model": "voyage-4"}, None), VoyageEmbedding)
