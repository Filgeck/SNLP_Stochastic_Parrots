import numpy as np
from datasets import load_dataset
import ollama
import random
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import time

# Load dataset, coding stack exchange q and a dataset 
ds = load_dataset("bigscience-data/roots_code_stackexchange")
print(f"Dataset loaded with {len(ds['train'])} examples")

NUM_SAMPLES = 10000  # this is about 1/100th of total set (9825059), but is faster
sample_indices = random.sample(range(len(ds['train'])), NUM_SAMPLES)
sampled_data = [ds['train'][i] for i in sample_indices]

# sentence transformer model for encoding
model_name = "all-MiniLM-L6-v2"  # model for embeddings, lightweight
encoder = SentenceTransformer(model_name)
print(f"Loaded encoder model: {model_name}")

# must have model running already via ollama cli for this bit to work
client = ollama.Client()

# questions from the dataset
corpus = []
total_len = len(sampled_data)
i = 0
for item in sampled_data:
    if i%(NUM_SAMPLES/10) == 0:
        print(f"Processing item {i}/{total_len}")
    i += 1
    text = item["text"]
    if "Q:\n\n" in text and "\n\nA:\n\n" in text:
        question = text.split("Q:\n\n")[1].split("\n\nA:\n\n")[0]
        answer = text.split("\n\nA:\n\n")[1]
        corpus.append({
            "question": question,
            "answer": answer,
            "full_text": text
        })

print(f"Extracted {len(corpus)} question-answer pairs")

# Encode corpus
print("Encoding corpus questions...")
start_time = time.time()
corpus_embeddings = encoder.encode([item["question"] for item in corpus], 
                                   show_progress_bar=True,
                                   convert_to_tensor=True)
print(f"Encoding completed in {time.time() - start_time:.2f} seconds")

# retrieve relevant documents
def retrieve_documents(query, k=3, similarity_metric="cosine"):
    """
    Retrieve the most similar documents to the query
    
    Args:
        query (str): The user query
        k (int): Number of documents to retrieve
        similarity_metric (str): Similarity measure to use
        
    Returns:
        list: Top k relevant documents
    """
    # encode the query
    query_embedding = encoder.encode(query, convert_to_tensor=True)
    
    # calculate similarity
    if similarity_metric == "cosine":
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        corpus_embeddings_np = corpus_embeddings.cpu().numpy()
        similarities = cosine_similarity(query_embedding_np, corpus_embeddings_np)[0]
        
    elif similarity_metric == "dot_product":
        # Calculate dot product
        similarities = torch.matmul(query_embedding, corpus_embeddings.T).cpu().numpy()
        
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
    
    # top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # top k documents
    retrieved_docs = [corpus[idx] for idx in top_k_indices]
    return retrieved_docs, similarities[top_k_indices]

def generate_response(query, retrieved_docs=None, model="llama3.2"):
    """
    Generate a response using Ollama
    
    Args:
        query (str): User query
        retrieved_docs (list, optional): Retrieved documents for RAG
        model (str): Model to use for generation
        
    Returns:
        str: Generated response
    """
    if retrieved_docs:
        # Construct RAG prompt with context
        context = "\n\n".join([f"QUESTION: {doc['question']}\nANSWER: {doc['answer']}" 
                              for doc in retrieved_docs])
        
        prompt = f"""You are a coding assistant. Use the following relevant StackOverflow posts 
        to help answer the user's question. Only use the information in these posts if relevant.
        
        RELEVANT STACKOVERFLOW POSTS:
        {context}
        
        USER QUESTION: {query}
        
        YOUR DETAILED ANSWER:"""
    else:
        # Regular prompt without RAG context
        prompt = f"""You are a coding assistant. Answer the following question to the best of your ability.
        
        USER QUESTION: {query}
        
        YOUR DETAILED ANSWER:"""
    
    # Generate response
    response = client.generate(model=model, prompt=prompt, stream=False)
    return response['response']

# Function to evaluate and compare
def compare_responses(query, k=3, similarity_metric="cosine", model="llama3.2"):
    """
    Compare responses with and without RAG
    
    Args:
        query (str): User query
        k (int): Number of documents to retrieve
        similarity_metric (str): Similarity measure to use
        model (str): Model to use for generation
        
    Returns:
        dict: Results with both responses and retrieved documents
    """
    # Retrieve relevant documents
    retrieved_docs, similarities = retrieve_documents(query, k, similarity_metric)
    
    print(f"Retrieved {len(retrieved_docs)} documents with similarities: {similarities}")
    
    # Generate responses
    print("Generating response without RAG...")
    response_without_rag = generate_response(query, None, model)
    
    print("Generating response with RAG...")
    response_with_rag = generate_response(query, retrieved_docs, model)
    
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "similarities": similarities.tolist(),
        "response_without_rag": response_without_rag,
        "response_with_rag": response_with_rag
    }

# Example query for testing
test_queries = [
    "How do I implement pagination in Django?",
    "What's the best way to handle authentication in a React application?",
    "How can I debug memory leaks in my Node.js application?",
    "What are the differences between lists and tuples in Python?",
    "How to use async/await with fetch in JavaScript?"
]

# Run the comparison for a selected query
if __name__ == "__main__":
    for selected_query in test_queries:

        print(f"Running comparison for query: '{selected_query}'")
        results = compare_responses(selected_query, k=3)
        
        # Print the results
        print("\n" + "="*50)
        print("QUERY:", results["query"])
        print("="*50)
        
        print("\nRETRIEVED DOCUMENTS:")
        for i, (doc, sim) in enumerate(zip(results["retrieved_documents"], results["similarities"])):
            print(f"\nDocument {i+1} (Similarity: {sim:.4f}):")
            print(f"Question: {doc['question'][:200]}...")
            print(f"Answer: {doc['answer'][:200]}...")
        
        print("\n" + "="*50)
        print("RESPONSE WITHOUT RAG:")
        print(results["response_without_rag"])
        
        print("\n" + "="*50)
        print("RESPONSE WITH RAG:")
        print(results["response_with_rag"])