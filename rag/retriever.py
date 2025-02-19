import logging
from langchain_aws import BedrockEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: BedrockEmbeddings | str,
                 client=None,
                 vector_store: InMemoryVectorStore | str | None = None,
                 kb_folder: str | None = None,
                 glob: str = '**/*.txt',
                 chunk_size: int = 500,
                 chunk_overlap: int = 100):
        if type(embedder) is BedrockEmbeddings:
            self.embeddings = embedder
        else:
            self.embeddings = BedrockEmbeddings(model_id=embedder, client=client)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if type(vector_store) is InMemoryVectorStore:
            self.vector_store = vector_store
        elif type(vector_store) is str:
            self.vector_store = InMemoryVectorStore.load(vector_store, self.embeddings)
        else:
            self.vector_store = InMemoryVectorStore(self.embeddings)
            if kb_folder is not None:
                self.__load_docs__(folder=kb_folder, glob=glob)

    def __load_docs__(self, folder: str, glob: str):
        loader = DirectoryLoader(folder, glob=glob, show_progress=True)
        docs = loader.load()
        all_splits = self.splitter.split_documents(docs)
        _ = self.vector_store.add_documents(documents=all_splits)

    def upload_file(self, filepath):
        logger.info(f"Uploading {filepath}...")
        name = Path(filepath).name
        with open(filepath) as f:
            content = f.read()
        logger.debug(content[:100])
        doc = Document(id=name, page_content=content, metadata={"extra": True, "source": name})
        all_splits = self.splitter.split_documents([doc])
        logger.debug(f"{len(all_splits)} splits created for {name}")
        _ = self.vector_store.add_documents(documents=all_splits)
        logger.info(f"Vector store updated with {name}.")
        return None

    def save_vector_store(self, file_path: str):
        self.vector_store.dump(file_path)

    def load_vector_store(self, file_path: str):
        self.vector_store = InMemoryVectorStore.load(file_path, self.embeddings)

    def embed(self, query: str):
        return self.embeddings.embed_query(query)

    def retrieve(self, query:str):
        return self.vector_store.similarity_search(query)
