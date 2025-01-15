import chromadb
from  utils.bge_emdedding_request import request_em

# 使用相同路径加载数据库
# client = chromadb.PersistentClient(path="/root/autodl-tmp/code/ai_rag_eval/chromadb")
#
# # 获取已保存的集合
# collection = client.get_collection(name="qiniu_doc")

def db_search(collection,text):
    # wenti
    # 查询数据
    em = request_em(([text]))[0]
    results = collection.query(
        query_embeddings=[em],
        n_results=5
    )
    return results
    # print(results)