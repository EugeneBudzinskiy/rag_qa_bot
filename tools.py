import config
from chain import Chain
from database import Database


class Tools:
    def __init__(self):
        self.database = Database()
        self.chain = Chain()

    def query_rag(self, question: str, history: list[dict[str, str]]) -> str:
        context = "no context"

        if self.chain.trigger_rag(question=question):

            for _ in range(config.MAX_TRANSFORM_QUESTION_ITERATIONS):
                query_result = self.database.collection.query(query_texts=question, n_results=config.RELEVANT_N_RESULTS)
                context = "\n\n".join(query_result["documents"][0])

                if self.chain.check_relevant(question=question, context=context):
                    break

                question = self.chain.transform_question(question=question, context=context, history=history)

        return self.chain.answer_question(question=question, context=context, history=history)
