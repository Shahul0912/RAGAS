# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # --- NEW: imports for Ragas ---
# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy, 
#     context_precision, 
#     context_recall
# )
# from datasets import Dataset

# # Step 1: Load PDF
# loader = PyPDFLoader("keys-to-trading-gold-ca.pdf")
# docs = loader.load()

# # Step 2: Split into chunks
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# split_docs = splitter.split_documents(docs)

# # Step 3: Create embeddings and vector store
# embedding_model = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(split_docs, embedding_model)

# # Step 4: Create retriever
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # Step 5: Setup chat memory
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Step 6: Create the Conversational Chain
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory,
#     verbose=True
# )

# # --- NEW: collect logs for Ragas evaluation ---
# questions, answers, contexts = [], [], []

# # Step 7: Chat loop
# while True:
#     query = input("Ask a question (or type 'exit'): ")
#     if query.lower() == "exit":
#         break

#     # Run the chain
#     result = qa_chain({"question": query})

#     # Extract the answer
#     answer = result["answer"]

#     # Extract the retrieved contexts
#     # (we can fetch directly from retriever to make sure we log them)
#     retrieved_docs = retriever.get_relevant_documents(query)
#     retrieved_texts = [doc.page_content for doc in retrieved_docs]

#     # Print response
#     print("Bot:", answer)

#     # Log for evaluation
#     questions.append(query)
#     answers.append(answer)
#     contexts.append(retrieved_texts)

# # --- NEW: After loop, run Ragas evaluation ---z
# if questions:
#     from datasets import Features, Value, Sequence
    
#     features = Features({
#         "question": Value("string"),
#         "answer": Value("string"),
#         "contexts": Sequence(Value("string"))
#     })

#     dataset = Dataset.from_dict({
#         "question": questions,
#         "answer": answers,
#         "contexts": contexts
#     }, features=features)

#     result = evaluate(
#         dataset,
#         metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
#     )

#     print("\n📊 Ragas Evaluation Results:")
#     print(result)


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()

# --- NEW: imports for Ragas ---
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy, 
    context_precision, 
    context_recall
)
from datasets import Dataset, Features, Value, Sequence

# Step 1: Load PDF
loader = PyPDFLoader("keys-to-trading-gold-ca.pdf")
docs = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Step 3: Create embeddings and vector store
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(split_docs, embedding_model)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 5: Setup chat memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 6: Create the Conversational Chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)

# --- NEW: collect logs for Ragas evaluation ---
questions, answers, contexts, ground_truths = [], [], [], []

# Step 7: Chat loop
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    # Run the chain
    result = qa_chain({"question": query})

    # Extract the answer
    answer = result["answer"]

    # Extract retrieved contexts
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # Print response
    print("Bot:", answer)

    # --- Log for evaluation ---
    questions.append(query)
    answers.append(answer)
    contexts.append(retrieved_texts)

    # For evaluation, you need a "ground truth" (reference answer)
    # Right now we'll just ask the user to provide it.
    gt = input("👉 Enter the correct answer (ground truth) for evaluation: ")
    ground_truths.append(gt)

# --- NEW: After loop, run Ragas evaluation ---
if questions:
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),
        "ground_truth": Value("string")
    })

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }, features=features)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    print("\n📊 Ragas Evaluation Results:")
    print(result)
