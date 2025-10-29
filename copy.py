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





os.environ["OPENAI_API_KEY"]=""

#load and split text file
loader=TextLoader()
document=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

full_content=format_docs(document)[:10000]


#embed and store chunks
embedding=OpenAIEmbedddings()
vectordb=Chroma.from_documents(chunks,embedding)
retriever=vectordb.as_retriever(search_kwargs={"k":3})

llm=ChatOpenAI(temperature=0.6)

#RAG prompt
template="""ou are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt=ChatPromptTemplate.from_template(template)

#ground truth prompt (strict to minimize hallucination
ground_truth_template = """You are an expert assistant. Generate a concise, accurate answer to the question based solely on the provided document context. Do not add external information, speculate, or elaborate beyond the context. If the context lacks sufficient information, state that clearly.
full document context: {context}
question: {question}
Ideal answer: """

ground_truth_prompt=ChatPromptTemplate.from_template(ground_truth_template)

#rag pipeline
chain=(
    {"context":retriever, "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

ground_truth_chain=ground_truth_prompt| llm | StrOutputParser()

#RagS dataset preparation
def prepare_ragas_dataset(question,context,answer,ground_truth):
    return Dataset.from_dict({
        "question": [question],
        "contexts": [contexts],
        "answer": [answer],
        "ground_truth": [ground_truth]
    })

#validate ground truth faithfulness
async def validate_ground_truth(question,groundtruth,full_context):
    dataset=Dataset.from_dict({
        "question":[question],
        "context":[[full_content]],
        "answer":[ground_truth]
    })
    result=evaluate(dataset=dataset, metrics=[faithfullness],llm=llm,embeddings=embeddings)
    return result["faithfullness"]

#evaluate function
async def evaluate_with_ragas(question, comtext, ground_truth,answer):
    gt_faithfullness = await validate_ground_truth(question, ground_truth, full_context)
    if gt_faithfullness <0.8:
        print("Warning: Ground truth may contain hallucinations (faithfulness: {:.3f}).".format(gt_faithfulness))

    dataset=prepare_ragas_dataset(question,context,answer,ground_truth)
    result=evaluate(
        dataset=dataset,
        metrucs=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )

    #extract scalar floats from lists
    scores={
        "faithfulness":result["faithfulness"][0],
        "answer_relevancy": result["answer_relevancy"][0],
        "context_precision": result["context_precision"][0],
        "context_recall": result["context_recall"][0]
    }
    return scores, gt_faithfullness

#main loop for real time evaluation

async def main():
    while True:
        query=input("ask a question ('or type exit'): ")
        if query.lower()=="exit":
            break
        #generate ground truth
        ground_truth=ground_truth_chain.invoke({"context":full_content, "question":query})
        print("Generated ground truth: ", ground_truth)
        #Run RAG
        answer = chain.invoke(query)
        print("RAG Answer: ", answer)
        contexts=[for doc in retriever.invoke(query)]
        #evaluate
        scores, gt_faithfulness= await evaluate_with_ragas(query, contexts, answer, ground_truth)
        print("\nScores: ")
        print(f"Ground Truth Faithfulness: {gt_faithfulness:.3f}")
        print(f"Faithfulness: {scores['faithfulness']:.3f}")
        print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
        print(f"Context Precision: {scores['context_precision']:.3f}")
        print(f"Context Recall: {scores['context_recall']:.3f}\n")



