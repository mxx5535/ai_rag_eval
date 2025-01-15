import requests

url = "http://localhost:8200/bge"
def request_em(context_list):


    payload = {"text": context_list}
    headers = {"content-type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers)
    embedding = eval(response.text)['embedding']
    print(response.content)
    return embedding

if __name__ == '__main__':
    request_em(["靠海吃海念海经”“沿海和山区都要树立全省‘一盘棋’的思想”等一系列重要发展理念"])