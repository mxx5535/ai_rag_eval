import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import FlagModel
import numpy as np




app = FastAPI()

model = None


class RequestTTS(BaseModel):  # 继承了BaseModel，定义了People的数据格式

    text: list = ["你好，欢迎使用变影系统"]




@app.on_event("startup")
def load_model():
    global model
    model = FlagModel('/root/autodl-tmp/model/bge',
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)
    print("model 加载完毕")




@app.post("/bge")
def postdeal(body: RequestTTS):  # 传入一个People类型的参数people
    try:
        print("model",model)
        print("body.text:",body.text)
        print("type:",type(body.text))
        embeddings_1 = model.encode(body.text)
        array_float32 = np.array(embeddings_1, dtype=np.float32)
        array_float = array_float32.astype(float).tolist()
        return {"num":len(embeddings_1),
                "embedding":array_float
                }
    except:
        pass





if __name__ == "__main__":
    uvicorn.run(app='bge_server:app', host='0.0.0.0', port=8200,workers=1)