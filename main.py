import pickle

import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import numpy as np
import chromadb
from utils.chdb_search import db_search
from langchain_openai import AzureChatOpenAI
import  os
from datasets import Dataset
from tqdm import tqdm
#
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test-az-eus-ai-openai01.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "02855675d52d4abfa48868c00c6f2773"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = 'test-az-eus-gpt-4o'
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"
#
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
#
# 使用相同路径加载数据库
client = chromadb.PersistentClient(path="/root/autodl-tmp/code/ai_rag_eval/chromadb")

# 获取已保存的集合
collection = client.get_collection(name="qiniu_doc")


# 构建数据集
eval_data = pd.read_csv('/root/autodl-tmp/code/ai_rag_eval/data/启牛话术文档rewrite.csv',index_col=0)
# 抽样
eval_data.reset_index(inplace=True)
eval_data = eval_data.rename(columns={"index":"type"})
eval_data_sample =eval_data.sample(n=50, random_state=42)
#
# ##
# questions = []
# ground_truths = []
# answers = []
# contexts = []
#
# for index,row in eval_data_sample.iterrows():
#     print('now:',index)
#     i = row['rewrite']
#     results = db_search(collection,i)
#     Q_list = results['documents'][0]
#     A_list = results['metadatas'][0]
#     content = []
#     for q,a in zip(Q_list,A_list):
#         content.append(q+'\t'+a['answer'])
#
#     template = """You are an assistant for question-answering tasks.
#     Use the following pieces of retrieved context to answer the question.
#     If you don't know the answer, just say that you don't know.
#
#     Question: {question}
#
#     Context: {context}
#
#     Answer:
#     """
#     prompt = template.format(question=i,context=contexts)
#     try:
#         eval_result = model.invoke(prompt).content
#     except:
#         continue
#     answers.append(eval_result)
#     contexts.append(content)
#     questions.append(row['rewrite'])
#     ground_truths.append([row['answer']])
#
# # 构建数据
# data = {
#     "question": questions,
#     "answer": answers,
#     "contexts": contexts,
#     "ground_truths": ground_truths
# }
# dataset = Dataset.from_dict(data)
# print(dataset)
#
# with open('dataset.pkl', 'wb') as file:
#     pickle.dump(dataset, file)
#
# with open('dataset.pkl', 'rb') as file:
#     dataset = pickle.load(file)
#
#
# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy
# )
# em_model = HuggingFaceBgeEmbeddings(model_name='/root/autodl-tmp/model/bge',model_kwargs ={'device':0},encode_kwargs = {'normalize_embeddings':True} )
# eval_result = evaluate(
#     dataset = dataset,
#     llm=model,
#     embeddings=em_model,
#     metrics=[
#         faithfulness,
#         answer_relevancy,
#     ],
# )
# print(eval_result)
# eval_result.to_pandas().to_csv('./eval_result.csv')


#-------检索系统----hit_rate- mrr----
def hit_rate_at_k(ground_truths, retrieved_docs, k=5):
    """
    计算 Hit Rate@k
    :param ground_truths: list of lists，包含每个查询的 ground truth 文档 ID
    :param retrieved_docs: list of lists，包含每个查询返回的 top-k 文档 ID
    :param k: 检索的前 k 个文档，默认为 5
    :return: hit rate at k
    """
    assert len(ground_truths) == len(retrieved_docs), "ground_truths 和 retrieved_docs 长度必须相同"

    hits = 0
    for truth, docs in zip(ground_truths, retrieved_docs):
        # 检查 ground truth 是否在前 k 个返回文档中
        if any(doc in docs[:k] for doc in truth):
            hits += 1

    hit_rate = hits / len(ground_truths)
    return hit_rate

ground_truths = []
retrieved_docs = []
mrr_score_list = []
for index,row in eval_data_sample.iterrows():
    ground_truths.append([row['type']])
    results = db_search(collection,row['question'])
    type_list = [i['type'] for i in results['metadatas'][0]]
    retrieved_docs.append(type_list)
    position = type_list.index(row['type'])+1
    mrr_score_list.append(1/position)

hit_rate_k = hit_rate_at_k(ground_truths,retrieved_docs)
mrr_score = np.mean(mrr_score_list)
print('hit_rate_k:',hit_rate_k)
print('mrr_score:',mrr_score)



