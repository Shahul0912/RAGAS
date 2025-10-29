import os
import asyncio
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Set OpenAI API key
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load and split text file
loader = TextLoader("/mnt/ddrive/RAG/illuminati.txt")
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(document)

# Format full context for ground truth (truncate to 10,000 characters)
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
full_context = format_docs(document)[:10000]  # Truncate to avoid context window limits

# Embed and store chunks
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(temperature=0.6)

# RAG prompt
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Ground truth prompt (strict to minimize hallucinations)
ground_truth_template = """You are an expert assistant. Generate a concise, accurate answer to the question based solely on the provided document context. Do not add external information, speculate, or elaborate beyond the context. If the context lacks sufficient information, state that clearly.
Full Document Context: {context}
Question: {question}
Ideal Answer:
"""
ground_truth_prompt = ChatPromptTemplate.from_template(ground_truth_template)

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ground truth chain
ground_truth_chain = ground_truth_prompt | llm | StrOutputParser()

# RAGAS dataset preparation
def prepare_ragas_dataset(question, contexts, answer, ground_truth):
    return Dataset.from_dict({
        "question": [question],
        "contexts": [contexts],
        "answer": [answer],
        "ground_truth": [ground_truth]
    })

# Validate ground truth faithfulness
async def validate_ground_truth(question, ground_truth, full_context):
    dataset = Dataset.from_dict({
        "question": [question],
        "contexts": [[full_context]],    #q list of all the list of retrieved chunks (which will be in a list format already)
        "answer": [ground_truth]
    })
    result = evaluate(dataset=dataset, metrics=[faithfulness], llm=llm, embeddings=embeddings)
    return result["faithfulness"][0]  # Extract scalar float from list

# Evaluation function
async def evaluate_with_ragas(question, contexts, answer, ground_truth):
    gt_faithfulness = await validate_ground_truth(question, ground_truth, full_context)
    if gt_faithfulness < 0.8:
        print("Warning: Ground truth may contain hallucinations (Faithfulness: {:.3f}).".format(gt_faithfulness))
    
    dataset = prepare_ragas_dataset(question, contexts, answer, ground_truth)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )
    # Extract scalar floats from lists
    scores = {
        "faithfulness": result["faithfulness"][0],
        "answer_relevancy": result["answer_relevancy"][0],
        "context_precision": result["context_precision"][0],
        "context_recall": result["context_recall"][0]
    }
    return scores, gt_faithfulness 

# Main loop for real-time evaluation
async def main():
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        # Generate ground truth
        ground_truth = ground_truth_chain.invoke({"context": full_context, "question": query})
        print("Generated Ground Truth:", ground_truth)
        # Run RAG
        answer = chain.invoke(query)
        print("RAG Answer:", answer)
        # Get contexts (use invoke instead of get_relevant_documents)
        contexts = [doc.page_content for doc in retriever.invoke(query)]
        # Evaluate
        scores, gt_faithfulness = await evaluate_with_ragas(query, contexts, answer, ground_truth)
        print("\nScores:")
        print(f"Ground Truth Faithfulness: {gt_faithfulness:.3f}")
        print(f"Faithfulness: {scores['faithfulness']:.3f}")
        print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
        print(f"Context Precision: {scores['context_precision']:.3f}")
        print(f"Context Recall: {scores['context_recall']:.3f}\n")

if __name__ == "__main__":
    asyncio.run(main())