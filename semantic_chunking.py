import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
pdf_path = "AI_Information.en.zh-CN.pdf"
API_KEY = ""

def extract_text_from_pdf(pdf_path):
    mypdf = fitz.open(pdf_path)
    all_text = ""  
    for page in mypdf:
        all_text += page.get_text("text") + " "
    return all_text.strip()


extracted_text = extract_text_from_pdf(pdf_path)
#print(extracted_text[:500])

client = ZhipuAI(api_key = API_KEY) 

def get_embedding(text: str):
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return np.array(response.data[0].embedding)

sentences = extracted_text.split('。')
print(f"Extracted {len(sentences)} sentences from the PDF.")

embeddings = [get_embedding(sentence) for sentence in sentences if sentence]
print(f"Generated {len(embeddings)} sentence embeddings.")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compute similarity between consecutive sentences
similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

def compute_breakpoints(similarities, method="percentile", threshold=90):
    # 根据选定的方法确定阈值
    if method == "percentile":
        # 计算相似度分数的第 X 百分位数
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # 计算相似度分数的均值和标准差。
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # 将阈值设置为均值减去 X 倍的标准差
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # 计算第一和第三四分位数（Q1 和 Q3）。
        q1, q3 = np.percentile(similarities, [25, 75])
        # 使用 IQR 规则（四分位距规则）设置阈值
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # 如果提供了无效的方法，则抛出异常
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # 找出相似度低于阈值的索引
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

# 使用百分位法计算断点，阈值为90
breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)

def split_into_chunks(sentences, breakpoints):
    chunks = []  # Initialize an empty list to store the chunks
    start = 0  # Initialize the start index

    # 遍历每个断点以创建块
    for bp in breakpoints:
        # 将从起始位置到当前断点的句子块追加到列表中
        chunks.append("。".join(sentences[start:bp + 1]) + "。")
        start = bp + 1  # 将起始索引更新为断点后的下一个句子

    # 将剩余的句子作为最后一个块追加
    chunks.append("。".join(sentences[start:]))
    return chunks  # Return the list of chunks

# split_into_chunks 函数创建文本块
text_chunks = split_into_chunks(sentences, breakpoints)

# Print the number of chunks created
print(f"Number of semantic chunks: {len(text_chunks)}")

# Print the first chunk to verify the result
print("\nFirst text chunk:")
print(text_chunks[0])

def create_embeddings(text_chunks):
    # Generate embeddings for each text chunk using the get_embedding function
    return [get_embedding(chunk) for chunk in text_chunks]

# Create chunk embeddings using the create_embeddings function
chunk_embeddings = create_embeddings(text_chunks)

def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    # 为查询生成嵌入
    query_embedding = get_embedding(query)

    # 计算查询嵌入与每个块嵌入之间的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # 获取最相似的 k 个块的索引
    top_indices = np.argsort(similarities)[-k:][::-1]

    # 返回最相关的 k 个文本块
    return [text_chunks[i] for i in top_indices]

# Load the validation data from a JSON file
with open('val.json', encoding="utf-8") as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Get top 2 relevant chunks
top_chunks = semantic_search(query, text_chunks, chunk_embeddings, k=2)

# Print the query
print(f"Query: {query}")

# Print the top 2 most relevant text chunks
for i, chunk in enumerate(top_chunks):
    print(f"Context {i+1}:\n{chunk}\n{'='*40}")

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
