import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

import config


class Chain:
    question_key: str = "question"
    history_key: str = "history"
    docs_key: str = "docs"

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            model=config.OPENAI_LANGUAGE_MODEL_NAME
        )

        self.handle_greeting_chain = (
            ChatPromptTemplate.from_template(template=config.HANDLE_GREETING_PROMPT)
            | self.llm
            | StrOutputParser()
        )

        self.response_greeting_chain = (
            ChatPromptTemplate.from_template(template=config.RESPONSE_GREETING_PROMPT)
            | self.llm
            | StrOutputParser()
        )

        self.doc_relevant_chain = (
            ChatPromptTemplate.from_template(template=config.DOC_RELEVANT_PROMPT)
            | self.llm
            | StrOutputParser()
        )

        self.transform_chain = (
            ChatPromptTemplate.from_template(template=config.TRANSFORM_PROMPT)
            | self.llm
            | StrOutputParser()
        )

        self.answer_chain = (
            ChatPromptTemplate.from_template(template=config.ANSWER_PROMPT)
            | self.llm
            | StrOutputParser()
        )

        self.fallback_chain = (
            ChatPromptTemplate.from_template(template=config.FALLBACK_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    def check_if_greeting(self, question: str) -> bool:
        answer = self.handle_greeting_chain.invoke({self.question_key: question})
        return answer.lower() == "true"

    def get_greeting_response(self, question: str) -> str:
        return self.response_greeting_chain.invoke({self.question_key: question})

    def check_if_relevant(self, question: str, docs: str) -> bool:
        answer = self.doc_relevant_chain.invoke({self.question_key: question, self.docs_key: docs})
        return answer.lower() == "true"

    def transform_question(self, question: str, docs: str, history: str) -> str:
        return self.transform_chain.invoke({
            self.question_key: question,
            self.docs_key: docs,
            self.history_key: history
        })

    def get_answer(self, question: str, docs: str, history: str) -> str:
        return self.answer_chain.invoke({
            self.question_key: question,
            self.docs_key: docs,
            self.history_key: history
        })

    def get_fallback(self, question: str) -> str:
        return self.fallback_chain.invoke({self.question_key: question})

