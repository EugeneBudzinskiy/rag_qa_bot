import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI

import config


class Chain:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            model=config.OPENAI_LANGUAGE_MODEL_NAME
        )

        trigger_rag_template = ChatPromptTemplate([
            ("system", config.TRIGGER_RAG_SYSTEM_PROMPT),
            ("user", "{question}")
        ])

        check_relevant_template = ChatPromptTemplate([
            ("system", config.CHECK_RELEVANT_SYSTEM_PROMPT),
            ("user", "{question}")
        ])

        transform_question_template = ChatPromptTemplate([
            ("system", config.TRANSFORM_QUESTION_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "{question}")
        ])

        answer_question_template = ChatPromptTemplate([
            ("system", config.ANSWER_QUESTION_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "{question}")
        ])

        self._trigger_rag_chain = (trigger_rag_template | self.llm | StrOutputParser())
        self._check_relevant_chain = (check_relevant_template | self.llm | StrOutputParser())
        self._transform_question_chain = (transform_question_template | self.llm | StrOutputParser())
        self._answer_question_chain = (answer_question_template | self.llm | StrOutputParser())

    def trigger_rag(self, question: str) -> bool:
        answer = self._trigger_rag_chain.invoke({
            "question": question
        })
        return answer.lower() == "true"

    def check_relevant(self, question: str, context: str) -> bool:
        answer = self._check_relevant_chain.invoke({
            "question": question,
            "context": context
        })
        return answer.lower() == "true"

    def transform_question(self, question: str, context: str, history: list[dict[str, str]]) -> str:
        return self._transform_question_chain.invoke({
            "question": question,
            "context": context,
            "history": history
        })

    def answer_question(self, question: str, context: str, history: list[dict[str, str]]) -> str:
        return self._answer_question_chain.invoke({
            "question": question,
            "context": context,
            "history": history
        })
