from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from schemas.embedding import HybridRetrievalResponse


def create_checkpoint_serializer() -> JsonPlusSerializer:
    """Create the serializer used by every Agent checkpoint backend."""

    return JsonPlusSerializer(
        allowed_msgpack_modules=[HybridRetrievalResponse],
    )
