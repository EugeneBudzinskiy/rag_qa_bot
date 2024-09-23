import config
from chain import Chain
from database import Database


class Tools:
    def __init__(self):
        self.database = Database()
        self.chain = Chain()

    def query_rag(self, query: str, history: str) -> str:
        if self.chain.check_if_greeting(question=query):
            return self.chain.get_greeting_response(question=query)

        for _ in range(config.MAX_TRANSFORM_QUESTION_ITERATIONS):
            query_result = self.database.collection.query(query_texts=query, n_results=config.RELEVANT_N_RESULTS)
            docs = "\n\n".join(query_result["documents"][0])

            if self.chain.check_if_relevant(question=query, docs=docs):
                break

            query = self.chain.transform_question(question=query, docs=docs, history=history)

        else:
            # Run out of iterations
            return self.chain.get_fallback(question=query)

        return self.chain.get_answer(question=query, docs=docs, history=history)
