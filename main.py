__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
import warnings

import chromadb
import streamlit as st

from PyPDF2 import PdfReader
from chromadb.utils import batch_utils
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chromadbx import UUIDGenerator
from langchain.text_splitter import RecursiveCharacterTextSplitter
# noinspection PyUnresolvedReferences
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_openai.chat_models import ChatOpenAI


warnings.simplefilter(action='ignore', category=FutureWarning)


PDF_FILE_PATH = "pdf/economic_evaluation_in_clinical_trials_henry_a_glick_jalpa_a_doshi-pages-0-46.pdf"
PAGES_TO_SKIP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

OPENAI_LANGUAGE_MODEL_NAME = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

CHROMA_PERSISTENT_PATH = "./db"
CHROMA_COLLECTION_NAME = "rag_test_collection"

TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 100
RELEVANT_N_RESULTS = 10

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context} 

Question: {question} 
"""


class Reader:
    @staticmethod
    def remove_header(text: str) -> str:
        header_match = re.search(pattern=r"\n", string=text)
        header_idx = header_match.span()[1] if header_match else 0
        if re.match(pattern=r"^\d+$", string=text[:header_idx]):
            header_match = re.search(pattern=r"\n", string=text[header_idx:])
            header_idx += header_match.span()[1] if header_match else 0
        return text[header_idx:]

    @staticmethod
    def remove_footer(text: str) -> str:
        footer_match = re.search(pattern=r"\n", string=text[::-1])
        footer_idx = footer_match.span()[1] if footer_match else -len(text)
        return text[:- footer_idx]

    @staticmethod
    def fix_paragraphs(text: str) -> str:
        return text.replace(".\n", ".\n\n")

    @staticmethod
    def fix_whitespaces(text: str) -> str:
        return (text
                .replace(" \n", " ")
                .replace("\xa0", " "))

    @staticmethod
    def fix_hyphen_usage(text: str) -> str:
        return (text
                .replace(" -\n", "")
                .replace("-\n", "-"))

    @staticmethod
    def remove_references(text: str) -> str:
        if re.search(pattern=r"^\s(\d)+\s", string=text):  # Is all page is reference list?
            return ""

        references_match = re.search(pattern="References", string=text)
        references_idx = references_match.span()[0] if references_match else len(text)
        return text[:references_idx]

    @classmethod
    def read_pdf(cls, path: str) -> str:
        result = ""
        with open(path, mode="rb") as f:
            reader = PdfReader(f)
            for i, page in enumerate(reader.pages):
                if i in PAGES_TO_SKIP:
                    continue

                raw_text = page.extract_text()

                raw_text = cls.remove_header(text=raw_text)
                raw_text = cls.remove_footer(text=raw_text)
                raw_text = cls.fix_whitespaces(text=raw_text)
                raw_text = cls.fix_paragraphs(text=raw_text)
                raw_text = cls.fix_hyphen_usage(text=raw_text)
                raw_text = cls.remove_references(text=raw_text)

                result += raw_text
        return result


class Tools:
    def __init__(self):
        self.chrome_client = chromadb.PersistentClient(path=CHROMA_PERSISTENT_PATH)

        self.llm_client = ChatOpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            model=OPENAI_LANGUAGE_MODEL_NAME
        )

        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=st.secrets["OPENAI_API_KEY"],
            model_name=OPENAI_EMBEDDING_MODEL_NAME
        )

        collection_exist_flag = CHROMA_COLLECTION_NAME in [x.name for x in self.chrome_client.list_collections()]

        self.collection = self.chrome_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function
        )

        if not collection_exist_flag:
            self.load_data_to_empty_chromadb()

    @staticmethod
    def get_text_chunks(text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_CHUNK_SIZE,
            chunk_overlap=TEXT_CHUNK_OVERLAP
        )
        return text_splitter.split_text(text)

    def save_to_chromadb(self, chunks: list[str]):
        batches = batch_utils.create_batches(
            api=self.chrome_client,
            documents=chunks,
            ids=UUIDGenerator(len(chunks))
        )
        for batch in batches:
            self.collection.add(*batch)

    def load_data_to_empty_chromadb(self):
        text = Reader.read_pdf(path=PDF_FILE_PATH)
        chunks = self.get_text_chunks(text=text)
        self.save_to_chromadb(chunks=chunks)

    def query_rag(self, query: str) -> str:
        query_result = self.collection.query(query_texts=query, n_results=RELEVANT_N_RESULTS)

        context = "\n\n".join(query_result["documents"][0])

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        chain = (
            prompt
            | self.llm_client
            | StrOutputParser()
        )
        return chain.invoke({"context": context, "question": query})


def main():
    tools = Tools()

    st.title("Simple Q&A Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = tools.query_rag(query=prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
