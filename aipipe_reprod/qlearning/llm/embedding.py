from collections import OrderedDict
from hashlib import md5
import json
import os
import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from loguru import logger


class EmbeddingsManager:

    def __init__(self, model_source='local', model_name='nomic-embed-text'):
        if model_source.lower() == 'local' or model_source.lower() == 'ollama':
            self.EMBEDDING_MODEL = OllamaEmbeddings(model=model_name)
        else:
            raise ValueError(f'Unexpected model source `{model_source}`!')
        
        index = faiss.IndexFlatL2(768)
        self.vector_store = FAISS(
            embedding_function=self.EMBEDDING_MODEL,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self._existing_fps = set()
        for doc in self.vector_store.docstore._dict.values():
            fp = self._fingerprint(doc)
            self._existing_fps.add(fp)

    def _fingerprint(self, text: str, meta: dict):
        meta_json = json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)
        raw = f'{text}||META||{meta_json}'
        return md5(raw.encode('utf-8')).hexdigest()

    def get_embedding_model(self):
        return self.EMBEDDING_MODEL

    def get_vector_store(self):
        return self.vector_store
    
    def load_vector_store(self, folder_path: str):
        self.vector_store = FAISS.load_local(folder_path, embeddings=self.EMBEDDING_MODEL, allow_dangerous_deserialization=True)
        if os.path.exists(f'{folder_path}/existing_fps.json'):
            with open(f'{folder_path}/existing_fps.json', 'r') as f:
                self._existing_fps = set(json.load(f))

    def save_vector_store(self, folder_path: str):
        self.vector_store.save_local(folder_path)
        with open(f'{folder_path}/existing_fps.json', 'w') as f:
            json.dump(list(self._existing_fps), f, ensure_ascii=False)

    def add_texts(self, texts: list[str], metadatas: list[dict] = None):
        to_add_text = []
        to_add_meta = []
        for text, meta in zip(texts, metadatas or []):
            fp = self._fingerprint(text, meta)
            if fp not in self._existing_fps:
                to_add_text.append(text)
                to_add_meta.append(meta)
                self._existing_fps.add(fp)
                
        if len(to_add_text) > 0 and len(to_add_meta) > 0:
            self.vector_store.add_texts(to_add_text, to_add_meta)

    def add_text(self, text: str, metadata: dict = None):
        fp = self._fingerprint(text, metadata)
        if fp not in self._existing_fps:
            self._existing_fps.add(fp)
            self.vector_store.add_texts([text], [metadata])
        else:
            logger.info(f'Already added: {fp}')

    def embed_text(self, text: str):
        return self.EMBEDDING_MODEL.embed_query(text)
    
    def embed_texts(self, texts: list[str]):
        return self.EMBEDDING_MODEL.embed_documents(texts)
    
    def search_by_vector(self, embedding: list[float], k: int = 5, filter: dict = None):
        return self.vector_store.similarity_search_by_vector(embedding, k, filter)
    
    def search_by_text(self, text: str, k: int = 5, filter: dict = None):
        return self.vector_store.similarity_search(text, k, filter)
    
    def search_by_text_dedup(self, text: str, k: int = 5, filter: dict = None):
        docs = self.vector_store.similarity_search(text, k * 5, filter)
        unique = list(OrderedDict((doc.id, doc) for doc in docs).values())
        return unique[:k]

if __name__ == '__main__':
    em = EmbeddingsManager()
    em.add_texts(['hello', 'world', '你好', '世界', '今天吃什么', '今天天气怎么样', '堵车了', 'hi'])
    query_results = em.search_by_text('hello', 3)
    for result in query_results:
        print(result.page_content, result.metadata)

