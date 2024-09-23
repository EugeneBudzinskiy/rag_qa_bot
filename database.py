import uuid

import chromadb
import streamlit as st
from chromadb.utils import batch_utils
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config
from reader import Reader


class Database:
    def __init__(self):
        self.reader = Reader()
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSISTENT_PATH)

        # noinspection PyUnresolvedReferences
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=st.secrets["OPENAI_API_KEY"],
            model_name=config.OPENAI_EMBEDDING_MODEL_NAME
        )

        self.collection_exist_flag = config.CHROMA_COLLECTION_NAME in [x.name for x in self.client.list_collections()]

        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function
        )

        if not self.collection_exist_flag:
            self.load_data_to_empty_chromadb()

    @staticmethod
    def get_text_chunks(text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.TEXT_CHUNK_SIZE,
            chunk_overlap=config.TEXT_CHUNK_OVERLAP
        )
        return text_splitter.split_text(text)

    def save_to_chromadb(self, chunks: list[str]):
        batches = batch_utils.create_batches(
            api=self.client,
            documents=chunks,
            ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
        )
        for batch in batches:
            self.collection.add(*batch)

    def load_data_to_empty_chromadb(self):
        text = self.reader.read_pdf(path=config.PDF_FILE_PATH)
        chunks = self.get_text_chunks(text=text)
        self.save_to_chromadb(chunks=chunks)
