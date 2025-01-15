import chromadb
# from chromadb.config import Settings
import pandas as pd
from utils.bge_emdedding_request import request_em



# 配置本地持久化存储路径
client = chromadb.PersistentClient(path="/root/autodl-tmp/code/ai_rag_eval/chromadb")
# 创建集合
collection = client.create_collection(name="qiniu_doc")

# 加载数据
data = pd.read_csv('/root/autodl-tmp/code/ai_rag_eval/data/启牛话术.csv')
for index ,row in data.iterrows():
    print("已到：",index)
    text_em = request_em([row['question']])[0]
    document_content = row['question']
    metadata = {"answer":row['answer'],"type":row['type']}
    collection.add(
        ids=[str(index)],
        documents=[document_content],
        embeddings=[text_em],
        metadatas=[metadata]
    )

# # 保存数据到本地
# client.persist()  # 持久化操作
