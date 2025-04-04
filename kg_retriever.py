from langchain_neo4j import Neo4jGraph
import math
from dotenv import dotenv_values

NEO4J_SECRETS = "neo4j_secrets.env"

dotenv_values(NEO4J_SECRETS).get("NEO4J_URL")

kg = Neo4jGraph(url=dotenv_values(NEO4J_SECRETS).get("NEO4J_URL") ,
                username=dotenv_values(NEO4J_SECRETS).get("NEO4J_USR"),
                password=dotenv_values(NEO4J_SECRETS).get("NEO4J_PWD"))

# Funzione che data una query crea il suo nodo e le relazioni con i concetti presenti in essa
def insertQueryNode(text, codes):
    cypher = """
        MERGE(q:Query {queryId: 'query0'})
        ON CREATE SET  
            q.text = $text,
            q.concepts = $concepts
        RETURN q    
    """
    kg.query(cypher, params={'text': text, 'concepts': codes})

    cypher = """
        MATCH (q:Query), (o:ObjectConcept)
            WHERE o.sctid = $concept
        MERGE (q)-[:HAS_CONCEPT]->(o)
    """
    for c in codes:
        kg.query(cypher, params={'concept': str(c)})

def get_chunk(id: str):
    return kg.query("MATCH (n:Chunk) WHERE n.chunkId = $id RETURN n", params={'id': id})[0]['n']

def shortest_path_id(id: str, max_hops: int = 10):
    chunk_ids = [c['id'] for c in kg.query("MATCH (n:Chunk) RETURN n.chunkId AS id")]

    cypher = f"""
        MATCH (start:ObjectConcept {{id: $id}}), (end:Chunk {{chunkId: $chunk_id}})
        MATCH path = shortestPath((start)-[*1..{max_hops}]-(end))
        WITH path
        WHERE all(r IN relationships(path) WHERE type(r) <> 'NEXT') AND all(n IN nodes(path)[0..-1] WHERE n:ObjectConcept)
        RETURN path
    """

    listPathChunks = list()
    for chunk_id in chunk_ids:
        paths = kg.query(cypher, params={'id': id, 'chunk_id': chunk_id})  # result è una lista di dizionari (contiene solo un dizionario con id path)
        if paths:
            path = paths[0]["path"]  # prendo il percorso (lista di dizionari (nodi) e stringhe (relazioni))
            node_count = math.ceil(len(path)/2)-2  # Distanza 0 = nodi direttamente collegati
            listPathChunks.append({"id": chunk_id, "path": path, "nodeCount": node_count})
    return sorted(listPathChunks, key=lambda x: x["nodeCount"])