from zhipuai import ZhipuAI
import fitz
import os
import numpy as np
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()
pdf_path = "AI_Information.en.zh-CN.pdf"
API_KEY = 


def extract_text_from_pdf(pdf_path):
    my_pdf = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(my_pdf.page_count):
        page = my_pdf[page_num]
        all_text += page.get_text("text")
    
    return all_text

client = ZhipuAI(api_key = API_KEY) 
# response = client.chat.completions.create(
#     model="glm-4",
#     messages=[
#         {"role": "user", "content": "你好"},
#     ],
# )
# print(response.choices[0].message)

def chunk_text(text, n, overlap):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])

    return chunks
    

extracted_text = extract_text_from_pdf(pdf_path)

text_chunks = chunk_text(extracted_text, 500, 100)

# print("Number of text chunks:", len(text_chunks))
# print("\n First text chunk:")
# print(text_chunks[0])


def create_embeddings(text):
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response

response = create_embeddings(text_chunks)

def cosine_similarity(vec1, vec2):
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = [] 

    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # Append the index and similarity score

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]

# Load the validation data from a JSON file
with open('val.json', encoding="utf-8") as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Perform semantic search to find the top 2 most relevant text chunks for the query
top_chunks = semantic_search(query, text_chunks, response.data, k=2)

# Print the query
print("Query:", query)

# Print the top 2 most relevant text chunks
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")

# Define the system prompt for the AI assistant
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

def generate_response(system_prompt, user_message):
    response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            top_p=0.8,
            #presence_penalty=1.05,
            max_tokens=4096,
        )
    return response.choices[0].message.content

# Create the user prompt based on the top chunks
user_prompt = "\n".join([f"上下文内容 {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\n问题: {query}"

# Generate AI response
ai_response = generate_response(system_prompt, user_prompt)
print(ai_response)

# Define the system prompt for the evaluation system
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

# Create the evaluation prompt by combining the user query, AI response, true response, and evaluation system prompt
evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# Generate the evaluation response using the evaluation system prompt and evaluation prompt
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)
