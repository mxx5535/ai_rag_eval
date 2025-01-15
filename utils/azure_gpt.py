import os

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test-az-eus-ai-openai01.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "02855675d52d4abfa48868c00c6f2773"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = 'test-az-eus-gpt-4o'
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"


from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
result = model.invoke("你好")
print(result.content)