# 上下文增强检索
import fitz
import numpy as np
import json
from zhipuai import ZhipuAI


pdf_path = "AI_Information.en.zh-CN.pdf"
API_KEY = "f532b4bd71324c1ca2fd7e22d4eb41da.4Sfpr8wE6qEVV5BY"


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


# 将文本分割为重叠的块
def chunk_text(text, n, overlap):
    chunks = []  
    for i in range(0, len(text), n - overlap):
        # 添加从当前索引到索引 + 块大小的文本块
        chunks.append(text[i:i + n])

    return chunks  # 返回文本块列表


# 设置客户端,处理文本
client = ZhipuAI(api_key = API_KEY) 

extract_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(extract_text, 1000, 200)

print("Number of text chunks:", len(text_chunks))
print(text_chunks[0])


# 嵌入
def create_embeddings(texts):
    # 确保每次调用不超过64条文本
    batch_size = 64
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="embedding-3",
            input=batch
        )
        # 将响应转换为numpy数组列表并添加到embeddings列表中
        embeddings.extend([np.array(embedding.embedding) for embedding in response.data])

    return embeddings

response = create_embeddings(text_chunks)


# 余弦相似度计算
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 上下文增强检索函数,找query的最相关文本块及其上下文
def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    # 将查询转换为嵌入向量
    query_embedding = create_embeddings(query)
    similarity_scores = []

    # 计算查询与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        # 计算查询嵌入与当前文本块嵌入之间的余弦相似度
        similarity_score = cosine_similarity(np.array(query_embedding), chunk_embedding)
        # 将索引和相似度分数存储为元组
        similarity_scores.append((i, similarity_score))

    # 按相似度分数降序排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关块的索引
    # top_index = [index for index, _ in similarity_scores[:k]]
    top_index = similarity_scores[0][0]

    # 定义上下文包含的范围
    # 确保不会超出 text_chunks 的边界
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)

    # 返回最相关的块及其相邻的上下文块
    return [text_chunks[i] for i in range(start, end)]

# 加载验证数据集
with open('val.json', encoding="utf-8") as val_file:
    data = json.load(val_file)

# 从数据集中提取第一个问题作为查询
query = data[0]['question']

# 检索最相关的块及其相邻的上下文块
# Parameters:
# - query: 检索的问题
# - text_chunks: 从 PDF 中提取的文本块
# - response.data: 文本块的嵌入
# - k=1: 返回最佳匹配
# - context_size=1: 最佳匹配上下文范围
top_chunks = context_enriched_search(query, text_chunks, response, k=1, context_size=1)

# Print the query for reference
print("Query:", query)
# 打印每个检索到的块，并附上标题和分隔符
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")

# 系统提示,定义回复
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

#根据系统、用户提示生成回复
def generate_response(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="glm-4",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

# 合并检索道德文本块
context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])

# 组合上下文和查询创建用户提示
user_prompt = f"{context}\n\nQuestion: {query}"
ai_response = generate_response(system_prompt, user_prompt)
print("AI Response:\n", ai_response)

# 评估响应质量
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 通过评估提示词评估系统
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)