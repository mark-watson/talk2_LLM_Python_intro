from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
loader = TextLoader("economics.txt")
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
llm = ChatOpenAI(model="gpt-4o-mini")

questions = [
    "Who says economics is bullshit?",
    "Who started the Austrian School of Economics?",
]

for query in questions:
    print("Query:", query)
    print("Answer:", index.query(query, llm=llm))

