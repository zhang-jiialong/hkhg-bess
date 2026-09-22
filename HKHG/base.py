from dataclasses import dataclass, field, asdict
from typing import TypedDict, Union, Literal, Generic, TypeVar
import asyncio

import numpy as np

# from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass



@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    concurrent_limit: int = 16

    def __post_init__(self):
        if self.concurrent_limit != 0:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        else:
            self._semaphore = UnlimitedSemaphore()

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        async with self._semaphore:
            return await self.func(*args, **kwargs)


@dataclass
class QueryParam:
    # use synonyms edge in planning and reasoning graph
    get_synonyms: bool = True
    # use target hyperedges as seed in planning and reasoning graph
    with_target_hyperedges: bool = True
    # include source text chunks as context for step answer and final answer generation
    with_src_chunks: bool = True
    # include plan context graph in plan initialisation
    with_plan_context: bool = True
    # include planning module
    with_planning: bool = True
    
    max_plan_depth: int = 3
    
    nb_of_init_plans: int = 2
    # ReasoningDAGAgent
    max_nb_of_answers: int = 2
    # DAG frontier search mode in reasoning
    search_tree_mode: Literal["DFS", "BFS"] = "DFS"

    # EOW based + LLM dir selection
    dir_selection: Literal["EOW", "ALL", "RAND"] = "ALL"
    # EW based daft path selection
    form_path_selection: Literal["EW", "RAND"] = "EW"

    # llm only / emb only / hybrid Entity Score Mode in reasoning
    score_mode: Literal["LLM", "EMB", "HYBRID"] = "HYBRID"
    # for HYBRID SCORE MODE
    emb_filter_threshold: float = 0.5
    # emb models in planning and reasoning
    emb_model: Literal["SBERT", "OPENAI"] = "OPENAI"

    # ReasoningStep
    max_paths: int = 20
    max_depth: int = 3
    max_width: int = 3
    draft_width: int = 20
    he_score_threshold: float = 0.6

    subquestion_guided: bool = False

    def print(self):
        """Pretty-print all configuration parameters in aligned text form."""
        data = asdict(self)
        key_width = max(len(k) for k in data)
        print("=" * (key_width + 25))
        print("QueryParam Configuration")
        print("=" * (key_width + 25))
        for k, v in data.items():
            print(f"{k:<{key_width}} : {v}")
        print("=" * (key_width + 25))



@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in HKHG.")
