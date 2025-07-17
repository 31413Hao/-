#上下文块标题（CCH）
import fitz
import numpy as np
from tqdm import tqdm
from zhipuai import ZhipuAI
import json


pdf_path = "AI_Information.en.zh-CN.pdf"
API_KEY = "f532b4bd71324c1ca2fd7e22d4eb41da.4Sfpr8wE6qEVV5BY"

# 设置模型
client = ZhipuAI(api_key = API_KEY) 
llm_model = "glm-4"
embedding_model="embedding-3"

# 提取pdf文本
def extract_text_from_pdf(pdf_path):
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串来存储提取的文本

    # 遍历PDF中的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text
    
    return all_text


# 使用LLM为给定的文本块生成标题/页眉
def generate_chunk_header(chunk):
    # 定义系统提示
    system_prompt = "为给定的文本生成一个简洁且信息丰富的标题。"

    # 根据系统提示和文本块生成
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )

    # 返回生成的标题/页眉
    return response.choices[0].message.content.strip()


# chunk+llm 生成标题
def chunk_text_with_headers(text, n, overlap):
    chunks = []

    # 按指定的块大小和重叠量遍历文本
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        header = generate_chunk_header(chunk)  # 使用 LLM 为块生成标题
        chunks.append({"header": header, "text": chunk})  # 将标题和块添加到列表中

    return chunks


extracted_text = extract_text_from_pdf(pdf_path)

# 把提取的文本分割为重叠的块，加上标题
text_chunks = chunk_text_with_headers(extracted_text, 1000, 200)

# 分块样例测试
print("Sample Chunk:")
print("Header:", text_chunks[0]['header'])
print("Content:", text_chunks[0]['text'])


# 为文本列表生成嵌入
def create_embeddings(texts):
    response = client.embeddings.create(
        model=embedding_model,
        input=texts
    )
    return response.data[0].embedding

# response = create_embeddings(text_chunks)

embeddings = []  # Initialize an empty list to store embeddings

# chunk嵌入显示
for chunk in tqdm(text_chunks, desc="Generating embeddings"):
    text_embedding = create_embeddings(chunk["text"])
    # print(text_embedding.shape)
    # 创建嵌入为块的标题
    header_embedding = create_embeddings(chunk["header"])
    # 
    embeddings.append({"header": chunk["header"], "text": chunk["text"], "embedding": text_embedding,
                       "header_embedding": header_embedding})
    
def cosine_similarity(vec1, vec2):
    # 余弦相似度计算
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 查询最相关的块
def semantic_search(query, chunks, k=5):
    query_embedding = create_embeddings(query)
    # print(query_embedding)
    # print(query_embedding.shape)

    similarities = []

    # 遍历每个块计算相似度分数
    for chunk in chunks:
        # 计算chunk和query之间的相似度
        sim_text = cosine_similarity(np.array(query_embedding), np.array(chunk["embedding"]))
        # sim_text = cosine_similarity(query_embedding, chunk["embedding"])

        # Compute cosine similarity between query embedding and chunk header embedding
        sim_header = cosine_similarity(np.array(query_embedding), np.array(chunk["header_embedding"]))
        # 计算平均相似度分数
        avg_similarity = (sim_text + sim_header) / 2
        # Append the chunk and its average similarity score to the list
        similarities.append((chunk, avg_similarity))

    # Sort the chunks based on similarity scores in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    # 返回最相关的k个
    return [x[0] for x in similarities[:k]]

with open('val.json', encoding="utf-8") as f:
    data = json.load(f)

query = data[2]['question']

# Retrieve the top 2 most relevant text chunks
top_chunks = semantic_search(query, embeddings, k=2)

# Print the results
print("Query:", query)
for i, chunk in enumerate(top_chunks):
    print(f"Header {i+1}: {chunk['header']}")
    print(f"Content:\n{chunk['text']}\n")


# AI 助手的系统提示
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

def generate_response(system_prompt, user_prompt):
    """
    基于检索到的文本块生成 AI 回答。

    Args:
    retrieved_chunks (List[str]): 检索到的文本块列表
    model (str): AI model.

    Returns:
    str: AI-generated response.
    """
    # Generate the AI response using the specified model
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Return the content of the AI response
    return response.choices[0].message.content

# 将检索到的文本块合并为一个上下文字符串
context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])

# 通过组合上下文和查询创建用户提示
user_prompt = f"{context}\n\nQuestion: {query}"
ai_response = generate_response(system_prompt, user_prompt)
print("AI Response:\n", ai_response)

evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 生成评估响应
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)