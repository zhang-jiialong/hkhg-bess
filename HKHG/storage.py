import asyncio
import html
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np


from collections import Counter

from nano_vectordb import NanoVectorDB

from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


def count_node_roles(graph: nx.Graph) -> dict[str, int]:
    """
    Counts the number of nodes with role 'entity' and 'hyperedge' in a NetworkX graph.
    Assumes all nodes have the 'role' attribute set to either 'entity' or 'hyperedge'.
    
    Parameters:
        graph: nx.Graph
            The NetworkX graph object.
    
    Returns:
        dict[str, int]: Dictionary with counts.
            Example:
            {
                "entity": 1500000,
                "hyperedge": 400000
            }
    """

    role_counter = {"entity": 0, "hyperedge": 0, "synonyms":0}
    for node, data in graph.nodes(data=True):
        if 'role' not in data:
            logger.info(f"{node} data does not contain 'role' attribute.")
            raise ValueError(f"{node} data must contain 'role' attribute.")
        role = data["role"]
        role_counter[role] += 1
    
    return {
        "entity": role_counter["entity"],
        "hyperedge": role_counter["hyperedge"],
        "synonyms": role_counter["synonyms"] 
    }





@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )
    
    async def get_vector_by_name(self, id: str):
        record = self._client.get([id])
        if record and "__vector__" in record[0]:
            return record[0]["__vector__"]
        logger.debug(f"id={id} not found in db.")
        return None
    

    async def get_vector_by_id(self, id: str):
        
        for i, data in enumerate(self.client_storage["data"]):           
            if data["__id__"] == id:
                vector = self.client_storage["matrix"][i]
                # logger.debug(f"Vector for id={id} found in db.")
                return vector
        logger.debug(f"id={id} not found in db.")
        return None
    
    async def query(self, query: str, top_k=5, better_than_threshold=None):
        if better_than_threshold is None:
            better_than_threshold=self.cosine_better_than_threshold
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    async def get_knn_by_id(self, id: str, top_k: int = 5, better_than_threshold=None):
        record = self._client.get([id])
        if not record or "__vector__" not in record[0]:
            return []
        vector = record[0]["__vector__"]
        return await self.get_knn_by_vector(
            vector=vector,
            top_k=top_k,
            better_than_threshold=better_than_threshold,
        )
    
    async def get_knn_by_vector(self, vector: np.ndarray, top_k: int = 5, better_than_threshold=None):
        if better_than_threshold is None:
            better_than_threshold = self.cosine_better_than_threshold
        knn = self._client.query(
            query=vector,
            top_k=top_k,
            better_than_threshold=better_than_threshold,
        )
        return [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in knn
        ]

    async def find_synonym(self, query: str, top_k=5, better_than_threshold=None, embedding=None):
        if embedding is None:
            embedding = await self.embedding_func([query])
            embedding = embedding[0]
        if better_than_threshold is None:
            better_than_threshold = self.cosine_better_than_threshold
        # else:
        #     better_than_threshold = float(better_than_threshold)
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )
    
    async def sample_random_entry(self):
        """
        Sample a random entry from the vector database.
        Returns a dictionary with the sampled entry's id and content.
        """
        if not self.client_storage["data"]:
            return None
        idx = np.random.randint(0, len(self.client_storage["data"]))
        entry = self.client_storage["data"][idx]
        return entry

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            node_role_count = count_node_roles(preloaded_graph)

            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {node_role_count['entity']} entities, {node_role_count['hyperedge']} hyperedges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
        self.edge_roles = ["link", "synonym", "similar", "contains_hyperedge"]

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)
    
    async def get_hyperedge(self, node_id: str) -> Union[dict, None]:
        data = self._graph.nodes.get(node_id)
        if data and data.get("role") == "hyperedge":
            return data
        return None

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str, data=False, role: str = "link"):
        if self._graph.has_node(source_node_id):
            if role in self.edge_roles:
                edges = []
                for node, target, edge_data in self._graph.edges(source_node_id, data=True):
                    if edge_data.get("role", "link") == role:
                        if data:
                            edges.append((node, target, edge_data))
                        else:
                            edges.append((node, target))


                return edges
            else:
                # Return all edges for the node
                return list(self._graph.edges(source_node_id, data=data))
        return None
    
    async def get_node_nbrs(self, source_node_id: str, data=False, role: str = "link"):
        if self._graph.has_node(source_node_id):
            if role in self.edge_roles:
                nbrs = []
                for _, target, edge_data in self._graph.edges(source_node_id, data=True):
                    if edge_data.get("role", "link") == role:
                        if data:
                            nbrs.append((target, edge_data))
                        else:
                            nbrs.append(target)
                return nbrs
            else:
                # Return all nbrs for the node
                return [[t[1:] for t in self._graph.edges(source_node_id, data=data)]]
        return None



    
    async def get_entity_synonyms(self, source_node_id: str):
        if self._graph.has_node(source_node_id) and self._graph.nodes.get(source_node_id).get("role") == "entity":
            
            group_edges = await self.get_node_nbrs(source_node_id, role="synonym")
            if not group_edges:
                return []
            group = group_edges[0]
            synonym_nbrs = await self.get_node_nbrs(group, role="synonym")

            synonyms = [entity for entity in synonym_nbrs if entity != source_node_id]
            return synonyms

        return None
    
    async def get_entity_hyperedges(self, source_node_id: str,  data=False, get_synonyms: bool = True):
        # print(self._graph.has_node(source_node_id))
        # print(self._graph.nodes.get(source_node_id).get("role") == "entity")
        
        if self._graph.has_node(source_node_id) and self._graph.nodes.get(source_node_id).get("role") == "entity":
            seen_hyperedges = set()
            hyperedges = []
            nodes = [source_node_id]
            if get_synonyms:
                synonyms = await self.get_entity_synonyms(source_node_id)
                if synonyms:
                    nodes.extend(synonyms)
            for node in nodes:
                for hyperedge in (await self.get_node_nbrs(node, data=data, role="link")):
                    # Ensure we only add unique hyperedges
                    if isinstance(hyperedge, tuple):
                        hyperedge_id = hyperedge[0]
                    else:
                        hyperedge_id = hyperedge
                    if hyperedge_id not in seen_hyperedges:
                        seen_hyperedges.add(hyperedge_id)
                        hyperedges.append(hyperedge)
            return hyperedges
        return None
    

    async def get_entity_similars(self, source_node_id: str):
        if self._graph.has_node(source_node_id) and self._graph.nodes.get(source_node_id).get("role") == "entity":
            similars = [entity for _, entity, _ in (await self.get_node_edges(source_node_id, data=True, role="similar"))]
            return similars
        return None
    
    async def get_hyperedge_entities(self, source_node_id: str, data: bool = False, get_synonyms: bool = True):
        if self._graph.has_node(source_node_id) and self._graph.nodes.get(source_node_id).get("role") == "hyperedge":
            entities = await self.get_node_nbrs(source_node_id, data=data, role="link")
            if not get_synonyms:
                return entities 
            all_entities = set()
            for entity in entities:
                synonyms = await self.get_entity_synonyms(entity)
                if synonyms:
                    all_entities.update(synonyms)
                all_entities.add(entity)
            return list(all_entities)
        return None
    
    async def get_entity_degree(self, source_node_id: str, mode: str = "hyperedge") -> int:
        """
        Returns hyperedge_count/ synonym_count/ similar_count for the given entity node.
        """
        if self._graph.has_node(source_node_id) and self._graph.nodes.get(source_node_id).get("role") == "entity":
            if mode == "hyperedge":
                edges = await self.get_entity_hyperedges(source_node_id, get_synonyms=False)
            elif mode == "hyperedge(get_synonyms)":
                edges = await self.get_entity_hyperedges(source_node_id, get_synonyms=True)
            elif mode == "synonym":
                edges = await self.get_entity_synonyms(source_node_id)
            elif mode == "similar":
                edges = await self.get_entity_similars(source_node_id)
            else:
                raise ValueError("Invalid mode. Choose from 'hyperedge', 'hyperedge(get_synonyms)', 'synonym', or 'similar'.")
            if edges:
                return len(edges)
        return 0
    
    async def get_hyperedge_degree(self, source_node_id: str) -> int:
        """
        Returns the number of entities linked to the given hyperedge node.
        """
        if self._graph.has_node(source_node_id) and self._graph.nodes.get(source_node_id).get("role") == "hyperedge":
            return self._graph.degree(source_node_id)
           
        return 0



    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
    
    def iter_entity_nodes(self):
        """
        Returns an iterator over (node_id, data) pairs for all nodes with role 'entity'.
        """
        return (
            (node_id, data)
            for node_id, data in self._graph.nodes(data=True)
            if data.get("role") == "entity"
        )
    
    def iter_hyperedge_nodes(self):
        """
        Returns an iterator over (node_id, data) pairs for all nodes with role 'hyperedge'.
        """
        return (
            (node_id, data)
            for node_id, data in self._graph.nodes(data=True)
            if data.get("role") == "hyperedge"
        )
    
    def iter_edges(self):
        """
        Returns an iterator over (source_node_id, target_node_id, data) for all edges in the graph.
        """
        return (
            (source, target, data)
            for source, target, data in self._graph.edges(data=True)
        )

    # async def statistics(self, level='synonym') -> dict:
    #     """
    #     Computes statistics:
    #     - total number of connected components
    #     - number of nodes and diameters of the top 100 largest components
    #     """



    #     G = self._graph
    #     # Degree distributions

    #     # for node_id, data in G.nodes(data=True):
    #     #     role = data.get("role")
    #     #     deg = G.degree(node_id)
    #     #     if role == "entity":
    #     #         entity_degrees.append(deg)
    #     #     elif role == "hyperedge":
    #     #         hyperedge_degrees.append(deg)
    #     entity_nodes = [node_id for node_id, data in G.nodes(data=True) if data.get("role") == "entity"]
    #     entity_degrees = await asyncio.gather(
    #         *[self.get_entity_degree(node_id, mode='hyperedge(get_synonyms)') for node_id in entity_nodes]
    #     )

    #     hyperedge_nodes = [node_id for node_id, data in G.nodes(data=True) if data.get("role") == "hyperedge"]
    #     hyperedge_degrees = await asyncio.gather(
    #         *[self.get_hyperedge_degree(node_id) for node_id in hyperedge_nodes]
    #     )

    #     # for node_id, data in G.nodes(data=True):
    #     #     role = data.get("role")
    #     #     if role == "entity":
    #     #         entity_degrees.append(len(await self.get_entity_hyperedges(node_id, get_synonyms=True)))
    #     #     elif role == "hyperedge":
    #     #         hyperedge_degrees.append(G.degree(node_id))

    #     entity_degree_dist = dict(Counter(entity_degrees))
    #     hyperedge_degree_dist = dict(Counter(hyperedge_degrees))

    #     entity_degree_dist_list = sorted(entity_degree_dist.items(), key=lambda x: x[0], reverse=True)
    #     hyperedge_degree_dist_list = sorted(hyperedge_degree_dist.items(), key=lambda x: x[0], reverse=True)

    #     logger.info(
    #         f"Entity degree distribution (dgr,cnt): {entity_degree_dist_list}, "
    #         f"Hyperedge degree distribution (dgr,cnt): {hyperedge_degree_dist_list}"
    #     )


    #     if level not in self.edge_roles:
    #         raise ValueError("Invalid level. Choose from 'link', 'synonym', or 'similar'.")
        
    #     if level == 'link':
    #         edges = [
    #             (u, v) for u, v, data in self._graph.edges(data=True)
    #             if data.get("role", 'link') == 'link'
    #         ]
    #         G = self._graph.edge_subgraph(edges).copy()
    #     elif level == 'synonym':
    #         edges = [
    #             (u, v) for u, v, data in self._graph.edges(data=True)
    #             if data.get("role", 'link') == 'link' or data.get("role", 'link') == 'synonym'
    #         ]
    #         G = self._graph.edge_subgraph(edges).copy()
        
    #     else:
    #         pass
    #         # Filter out only edges with role="link" or "synonym"

    #     if G.is_directed():
    #         components = list(nx.weakly_connected_components(G))
    #     else:
    #         components = list(nx.connected_components(G))

    #     # Sort components by size (largest first)
    #     components.sort(key=len, reverse=True)

    #     size_list = []
    #     diameters_list = []

    #     for comp_nodes in components[:100]:
    #         subgraph = G.subgraph(comp_nodes)
    #         node_role_count = count_node_roles(subgraph)
    #         size_list.append((node_role_count['entity'], node_role_count['hyperedge']))
            
            
    #         # if node_role_count["entity"] == 1:
    #         #     diameter = 0
    #         # else:
    #         #     try:
    #         #         diameter = nx.diameter(subgraph)
    #         #     except nx.NetworkXError:
    #         #         diameter = None
    #         # diameters_list.append(diameter)

    #     logger.info(
    #         f"Total number of connected components: {len(components)}, "
    #         f"Top 100 component sizes (entities, hyperedges): {size_list}"
    #     )
        
    #     return {
    #         "total_number_of_components": len(components),
    #         "top_100_component_stats": {
    #             "component_size(entities,hyperedges)": size_list,
    #             # "diameters": diameters_list, 
    #         },
    #         "entity_degree_distribution": entity_degree_dist_list,
    #         "hyperedge_degree_distribution": hyperedge_degree_dist_list,
    #     }
    
    def get_similar_components(self) -> list[set]:
        # Filter out only edges with role="similar"
        similar_edges = [
            (u, v) for u, v, data in self._graph.edges(data=True)
            if data.get("role") == "similar"
        ]
        # Create a subgraph with only "similar" edges
        similar_subgraph = self._graph.edge_subgraph(similar_edges).copy()
        components = list(nx.connected_components(similar_subgraph))

        return components
    
