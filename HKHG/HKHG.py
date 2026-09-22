import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast
import numpy as np

from .llm import (
    gpt_4o_mini_complete,
    openai_compatible_complete,
    openai_embedding,
    gpt_35_turbo,
    gpt_4_turbo,
    qwen3_8b,
    make_sentence_transformer_embedding_func,
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    build_higher_order_hyperedges,
    merge_synonym_entities,
    kg_query_with_reasoning,
)

from .utils import (
    # EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
    split_string_by_multi_markers,
    # array_to_buffer_string, 
    # buffer_string_to_array,
)
from .prompt import GRAPH_FIELD_SEP
from .base import (
    EmbeddingFunc,
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name, package=package)

        # Get the class from the module and instantiate it
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class HKHG:
    working_dir: str = field(
        default_factory=lambda: f"hkhg_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = field(default="gpt-4o-mini") #"meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'

    llm_model_max_token_size: int = 32768
    llm_input_token_safety_margin: int = 512
    # llm_model_max_async: int = 16
    llm_model_max_async: int = field(default=20)
    llm_model_kwargs: dict = field(
        default_factory=lambda: {
            "base_url": os.environ.get("OPENAI_BASE_URL"),
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    )
    max_concurrency: int = field(default=20)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = field(default="gpt-4o-mini")
    # tiktoken_model_name: str = field(default="gpt-3.5-turbo")

    # entity extraction
    entity_extract_max_gleaning: int = 2
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_model_name: str | None = "text-embedding-3-large"
    embedding_model_device: str | None = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 20



    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    similar_params: dict = field(default_factory=lambda: {
        "similar_threshold": 0.9,
        "similar_top_k": 10,
        # "similar_embedding_func": openai_embedding,
        # "similar_embedding_batch_num": 32,
        # "similar_embedding_max_async": 16,
    })
    higher_order_params: dict = field(default_factory=lambda: {
        "enabled": True,
        "max_level": 4,
        "cluster_strategy": "mixed_graph_community",
        "semantic_top_k": 30,
        "semantic_ef_search": 120,
        "semantic_hnsw_m": 16,
        "semantic_pca_dim": 256,
        "higher_order_batch_size": 8,
        "topology_hops": 2,
        "semantic_weight": 0.7,
        "topology_weight": 0.3,
        "topology_direct_weight": 0.5,
        "topology_path_weight": 0.5,
        "topology_path_decay": 0.7,
        "mixed_similarity_threshold": 0.5,
        "min_cluster_size": 2,
        "min_samples": 1,
        "cluster_metric": "cosine",
        "cluster_selection_epsilon": 0.0,
        "max_summary_clusters": None,
        "max_cluster_size": 100,
        "promote_singletons": True,
        "skip_if_exists": True,
    })

    log_file: str = field(default="proh.log")
    cosine_better_than_threshold: float = field(default=0.2)

    def __post_init__(self):

        set_logger(self.log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"HKHG init with param:\n  {_print_config}\n")

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        if self.embedding_model_name:
            if os.path.exists(self.embedding_model_name):
                embedding_device = self.embedding_model_device
                if embedding_device is None:
                    try:
                        import torch

                        embedding_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    except Exception:
                        embedding_device = "cpu"
                    self.embedding_model_device = embedding_device
                logger.info(
                    f"Loading local embedding model from {self.embedding_model_name}"
                    + (
                        f" on device {embedding_device}"
                        if embedding_device
                        else ""
                    )
                )
                self.embedding_func = make_sentence_transformer_embedding_func(
                    self.embedding_model_name,
                    device=embedding_device,
                )
            else:
                logger.info(
                    f"Using OpenAI-compatible embedding endpoint for model {self.embedding_model_name}."
                )

                async def _remote_embedding(texts: list[str]) -> np.ndarray:
                    return await openai_embedding.func(
                        texts,
                        model=self.embedding_model_name,
                        base_url=self.llm_model_kwargs.get("base_url"),
                        api_key=self.llm_model_kwargs.get("api_key"),
                        timeout=self.llm_model_kwargs.get("timeout"),
                    )

                embedding_dim = 3072 if self.embedding_model_name == "text-embedding-3-large" else 1536
                self.embedding_func = EmbeddingFunc(
                    embedding_dim=embedding_dim,
                    max_token_size=openai_embedding.max_token_size,
                    func=_remote_embedding,
                )
        
        model_map = {
            "gpt-4o-mini": (gpt_4o_mini_complete, "gpt-4o-mini"),
            "gpt-3.5-turbo": (gpt_35_turbo, "gpt-3.5-turbo"),
            "gpt-4-turbo": (gpt_4_turbo, "gpt-4-turbo-2024-04-09"),
            "qwen3-8b": (qwen3_8b, "qwen3-8b"),
        }

        if self.llm_model_name in model_map:
            self.llm_model_func, self.tiktoken_model_name = model_map[self.llm_model_name]
        elif self.llm_model_kwargs.get("base_url") or self.llm_model_kwargs.get("api_key"):
            logger.info(
                f"Using OpenAI-compatible endpoint for custom model {self.llm_model_name}."
            )
            self.llm_model_func = partial(openai_compatible_complete, self.llm_model_name)
            self.tiktoken_model_name = "gpt-4o-mini"
            if self.llm_model_name.lower().startswith("qwen") and self.llm_model_max_token_size > 16384:
                logger.info(
                    "Clamping llm_model_max_token_size to 16384 for qwen-compatible endpoint."
                )
                self.llm_model_max_token_size = 16384
        else:
            logger.info(f"Unknown model name: {self.llm_model_name}. Falling back to gpt-4o-mini.")
            self.llm_model_name = "gpt-4o-mini"
            self.llm_model_func, self.tiktoken_model_name = model_map[self.llm_model_name]
        logger.info(f"LLM model set to {self.llm_model_name} with tiktoken model {self.tiktoken_model_name}")
        
      



        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )


        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.hyperedges_vdb = self.vector_db_storage_cls(
            namespace="hyperedges",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"hyperedge_name"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

        self.entity_names_vdb = self.vector_db_storage_cls(
            namespace="entity_names",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )


    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }
    
    def get_graph_statistics(self, level='synonym') -> dict:
        knowledge_graph_inst=self.chunk_entity_relation_graph
        if knowledge_graph_inst is None:
            return {}
        loop = always_get_an_event_loop()
        return loop.run_until_complete(knowledge_graph_inst.statistics(level=level))
    

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))
    

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                hyperedge_vdb=self.hyperedges_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new hyperedges and entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
            logger.info("[Higher-Order Hyperedge Construction]...")
            self.chunk_entity_relation_graph = await build_higher_order_hyperedges(
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                text_chunks_db=self.text_chunks,
                entity_vdb=self.entities_vdb,
                hyperedge_vdb=self.hyperedges_vdb,
                global_config=asdict(self),
            )
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            # self.entity_name_emb_cache,
            self.entity_names_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.hyperedges_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.hyperedges_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query_reasoning(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_reasoning(query, param))

    async def aquery_reasoning(self, query: str, param: QueryParam = QueryParam()):
        response = await kg_query_with_reasoning(
            query,
            knowledge_graph_inst= self.chunk_entity_relation_graph,
            entities_vdb = self.entities_vdb,
            entity_names_vdb = self.entity_names_vdb,
            hyperedges_vdb = self.hyperedges_vdb,
            text_chunks_db = self.text_chunks,
            query_param = param,
            global_config = asdict(self),
            hashing_kv=self.llm_response_cache,
        )
        return response
    


    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.hyperedges_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    
    

    async def ainit_entity_names_vdb(self):
        entity_name_upsert_data = {}
        for entity_name, entity_data in self.chunk_entity_relation_graph.iter_entity_nodes():
            id = compute_mdhash_id(entity_name, prefix="en-")
            entity_name_upsert_data[id] = {
                "content": entity_name,
                "entity_name": entity_name,
            }
        if entity_name_upsert_data:
            await self.entity_names_vdb.upsert(entity_name_upsert_data)

    def init_entity_names_vdb(self):
        logger.info("Initializing entity names vector database...")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainit_entity_names_vdb())

    async def _areset_entity_names_vdb(self):
        if self.entity_names_vdb is not None:
            try:
                os.remove(self.entity_names_vdb._client_file_name)
            except FileNotFoundError:
                pass
        self.entity_names_vdb = self.vector_db_storage_cls(
            namespace="entity_names",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )

    async def _aclear_similarity_and_synonym_artifacts(self):
        graph = getattr(self.chunk_entity_relation_graph, "_graph", None)
        if graph is None:
            return

        removable_edges = [
            (source, target)
            for source, target, edge_data in graph.edges(data=True)
            if edge_data.get("role") in {"similar", "synonym"}
        ]
        if removable_edges:
            graph.remove_edges_from(removable_edges)

        synonym_nodes = [
            node_id
            for node_id, node_data in graph.nodes(data=True)
            if node_data.get("role") == "synonyms"
        ]
        for node_id in synonym_nodes:
            graph.remove_node(node_id)

        logger.info(
            f"Cleared {len(removable_edges)} similar/synonym edges and {len(synonym_nodes)} synonym group nodes."
        )

    async def _aclear_higher_order_artifacts(self):
        higher_order_nodes = []
        higher_order_chunk_ids = set()
        generated_entities = set()

        for node_id, node_data in list(self.chunk_entity_relation_graph.iter_hyperedge_nodes()):
            if int(node_data.get("hyperedge_level", 0) or 0) < 1:
                continue
            higher_order_nodes.append(node_id)
            source_ids = split_string_by_multi_markers(
                node_data.get("source_id", ""),
                [GRAPH_FIELD_SEP],
            )
            higher_order_chunk_ids.update(source_id for source_id in source_ids if source_id)
            generated = split_string_by_multi_markers(
                node_data.get("generated_entities", ""),
                [GRAPH_FIELD_SEP],
            )
            generated_entities.update(entity_name for entity_name in generated if entity_name)

        for node_id in higher_order_nodes:
            await self.chunk_entity_relation_graph.delete_node(node_id)

        for chunk_id in higher_order_chunk_ids:
            self.text_chunks._data.pop(chunk_id, None)

        if higher_order_nodes:
            higher_order_relation_ids = [
                compute_mdhash_id(node_id, prefix="rel-")
                for node_id in higher_order_nodes
            ]
            self.hyperedges_vdb._client.delete(higher_order_relation_ids)

        stale_entities = []
        for entity_name in generated_entities:
            entity_data = await self.chunk_entity_relation_graph.get_node(entity_name)
            if not entity_data or entity_data.get("role") != "entity":
                continue

            current_source_ids = set(
                split_string_by_multi_markers(
                    entity_data.get("source_id", ""),
                    [GRAPH_FIELD_SEP],
                )
            )
            remaining_source_ids = sorted(
                source_id
                for source_id in current_source_ids
                if source_id and source_id not in higher_order_chunk_ids
            )
            link_edges = await self.chunk_entity_relation_graph.get_node_edges(
                entity_name,
                data=True,
                role="link",
            )
            if not remaining_source_ids and not link_edges:
                stale_entities.append(entity_name)
                continue
            if remaining_source_ids != sorted(source_id for source_id in current_source_ids if source_id):
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name,
                    node_data={
                        **entity_data,
                        "source_id": GRAPH_FIELD_SEP.join(remaining_source_ids),
                    },
                )

        for entity_name in stale_entities:
            await self.entities_vdb.delete_entity(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

        logger.info(
            f"Cleared {len(higher_order_nodes)} higher-order hyperedges, {len(higher_order_chunk_ids)} higher-order chunks, and {len(stale_entities)} orphan generated entities."
        )

    def rebuild_higher_order_layers(
        self,
        refresh_similar_edges: bool = True,
        refresh_synonyms: bool = True,
    ):
        logger.info("Rebuilding higher-order hyperedges from existing L0 graph...")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.arebuild_higher_order_layers(
                refresh_similar_edges=refresh_similar_edges,
                refresh_synonyms=refresh_synonyms,
            )
        )

    async def arebuild_higher_order_layers(
        self,
        refresh_similar_edges: bool = True,
        refresh_synonyms: bool = True,
    ):
        await self._aclear_similarity_and_synonym_artifacts()
        await self._aclear_higher_order_artifacts()
        await self._areset_entity_names_vdb()

        original_skip_if_exists = self.higher_order_params.get("skip_if_exists", True)
        self.higher_order_params["skip_if_exists"] = False
        try:
            self.chunk_entity_relation_graph = await build_higher_order_hyperedges(
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                text_chunks_db=self.text_chunks,
                entity_vdb=self.entities_vdb,
                hyperedge_vdb=self.hyperedges_vdb,
                global_config=asdict(self),
            )
        finally:
            self.higher_order_params["skip_if_exists"] = original_skip_if_exists

        await self.ainit_entity_names_vdb()
        await self._insert_done()

        if refresh_similar_edges:
            await self.aadd_similar_edges()
        if refresh_synonyms:
            await self.amerge_synonyms()
    

    def add_similar_edges(self):
        logger.info("Adding similar edges to the knowledge graph...")
        logger.info(f"similar_params: top_k={self.similar_params['similar_top_k']}, better_than_threshold={self.similar_params['similar_threshold']}")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aadd_similar_edges())
    
    async def aadd_similar_edges(self):
        """
        Add similar edges to the knowledge graph based on the entities in the vector database.
        This function retrieves all entities, checks for similars, and creates edges between them.
        """
        
        for source, target, data in self.chunk_entity_relation_graph.iter_edges():
            # if role does not exist, create it
            if "role" not in data:
                await self.chunk_entity_relation_graph.upsert_edge(
                    source,
                    target,
                    edge_data={
                        **data,
                        "role": "link",
                    }
                )
        
        similar_edge_count = 0
        if self.entity_names_vdb is None or self.chunk_entity_relation_graph is None:
            logger.warning("Vector DB or Knowledge Graph is not initialized.")
            return
        
        try:
            for entity_name, entity_data in self.chunk_entity_relation_graph.iter_entity_nodes():
                id = compute_mdhash_id(entity_name, prefix="en-")
                entity_name_emb = await self.entity_names_vdb.get_vector_by_id(id)
                if entity_name_emb is None:
                    logger.debug(f"Entity {entity_name} id={id} not found in entity_names_vdb, upserting it.")

                    await self.entity_names_vdb.upsert(
                        {id: {"content": entity_name, "entity_name": entity_name}}
                    )
                    entity_name_emb = await self.entity_names_vdb.get_vector_by_id(id)
                if entity_name_emb is None:
                    logger.warning(f"Entity '{entity_name}' has no embedding, skipping.")
                    continue

                # similars = await self.entity_names_vdb.get_knn_by_id(
                #     id=id, 
                #     top_k=self.similar_params["similar_top_k"],
                #     better_than_threshold=self.similar_params["similar_threshold"],
                #     )
                similars = await self.entity_names_vdb.get_knn_by_vector(
                        vector=entity_name_emb,
                        top_k=self.similar_params["similar_top_k"],
                        better_than_threshold=self.similar_params["similar_threshold"],
                    )
                
                if not len(similars):
                    continue
                if len(similars) > 1:
                    logger.debug(f"Found {len(similars)} similars for entity '{entity_name}'")
                # create edges between the query entity_name and its similars

                for similar in similars:
                    similar_name = similar["entity_name"]
                    distance = round(similar["distance"],4)
                    if similar_name == entity_name:
                        continue
                    # check if the similar already exists in the knowledge graph
                    if await self.chunk_entity_relation_graph.has_node(similar_name):
                        # create an edge between the query entity_name and the similar
                        await self.chunk_entity_relation_graph.upsert_edge(
                            entity_name,
                            similar_name,
                            edge_data={
                                "weight": distance,
                                "role": "similar",
                            }
                        )
                        similar_edge_count += 1
                        logger.debug(f"Added similar edge {distance:.4f} from {entity_name} to {similar_name}")

            logger.info(f"{similar_edge_count} similar edges added successfully.")
        except Exception as e:
            logger.error(f"Error while adding similar edges: {e}")
        finally:
            # if update_storage: # TODO: check if this is needed
            await self._insert_done()

    def merge_synonyms(self, max_compare_batch_size=20, max_shuffle_attempts=3):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_synonyms(
                max_compare_batch_size=max_compare_batch_size,
                max_shuffle_attempts=max_shuffle_attempts,
            )
        )
    
    async def amerge_synonyms(
        self, max_compare_batch_size=20, max_shuffle_attempts=3
    ):
        await merge_synonym_entities(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.hyperedges_vdb,
            asdict(self),
            max_compare_batch_size=max_compare_batch_size,
            max_shuffle_attempts=max_shuffle_attempts,
        )
        await self._insert_done()
    


    # def supplement(self, segments):
    #     loop = always_get_an_event_loop()
    #     return loop.run_until_complete(self.asupplement(segments))
    

    # async def asupplement(self, segments):
    #     try:
            
    #         if not isinstance(segments, list):
    #             segments = [segments]

    #         if not segments:
    #             return

    #         logger.info(f"[Supplement] inserting {len(segments)} segments")

    #         cur_segments = segments
    #         inserting_segments = {}


    #         logger.info("[Entity Extraction]...")
    #         while cur_segments:
    #             pending = []
    #             for source_text_chunk_id, segment in cur_segments:
    #                 segment = segment.strip()
    #                 if source_text_chunk_id in inserting_segments:
    #                     pending.append((source_text_chunk_id, segment))
    #                     continue
    #                 inserting_segments[source_text_chunk_id] = {"content": segment}
   
    #             maybe_new_kg = await extract_entities(
    #                 inserting_segments,
    #                 knowledge_graph_inst=self.chunk_entity_relation_graph,
    #                 entity_vdb=self.entities_vdb,
    #                 hyperedge_vdb=self.hyperedges_vdb,
    #                 global_config=asdict(self),
    #             )
    #             if maybe_new_kg is None:
    #                 logger.warning("No new hyperedges and entities found")
    #                 continue
    #             self.chunk_entity_relation_graph = maybe_new_kg
    #             cur_segments = pending
    #             inserting_segments = {}
    #     finally:
    #             await self._insert_done()

