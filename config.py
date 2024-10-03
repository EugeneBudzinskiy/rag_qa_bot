PDF_FILE_PATH = "pdf/economic_evaluation_in_clinical_trials_henry_a_glick_jalpa_a_doshi-pages-0-46.pdf"
PAGES_TO_SKIP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

OPENAI_LANGUAGE_MODEL_NAME = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

CHROMA_PERSISTENT_PATH = "./db"
CHROMA_COLLECTION_NAME = "rag_test_collection"

TEXT_CHUNK_SIZE = 600
TEXT_CHUNK_OVERLAP = 200
RELEVANT_N_RESULTS = 5

MAX_TRANSFORM_QUESTION_ITERATIONS = 3


TRIGGER_RAG_SYSTEM_PROMPT = """
You are an part of internal system for assistant specializing in healthcare economic evaluation, particularly in 
clinical trials. Your task is to determine whether is user input is the question or not.
If the user's input is a question: return True, otherwise return: False.
"""


CHECK_RELEVANT_SYSTEM_PROMPT = """
You are an part of internal system for assistant specializing in healthcare economic evaluation, particularly in 
clinical trials. You are tasked with determining whether the retrieved documents are relevant to the user's question. 
If the retrieved documents are relevant to the original question return: True, otherwise return: False
The system has retrieved some documents based on the user's question: 


{context}
"""


TRANSFORM_QUESTION_SYSTEM_PROMPT = """
You are an part of internal system for assistant specializing in healthcare economic evaluation, particularly in 
clinical trials. The original question posed by the user has no directly relevant documents. Using the chat history 
and the retrieved, though irrelevant, documents, you need to reformulate the original question while preserving its 
original intent. Transform the user's original question into a new version that might yield relevant results. 
Incorporate any helpful information from previous exchanges and retrieved documents, even if they aren’t fully 
relevant, to maintain coherence.


{context}
"""


ANSWER_QUESTION_SYSTEM_PROMPT = """
You are a highly knowledgeable assistant specializing in economic evaluation in healthcare, with a focus on clinical 
trials. Utilize the following retrieved context to provide a concise and precise answer. Highlight key methodological 
and practical aspects where relevant. If the context does not directly answer the question, say that in polite manner. 
Maintain brevity with no more than three sentences, emphasizing clarity and practicality.


{context}
"""
