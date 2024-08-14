import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize components
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
loader = TextLoader("economics.txt")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Create and save index
docs = loader.load_and_split()
texts = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(texts, embedding)
vectorstore.save_local("faiss_index")

# Reload index
loaded_vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=loaded_vectorstore.as_retriever()
)

# Perform queries
questions = [
  "Who says economics is bullshit?",
  "Who started the Austrian School of Economics?",
]

for query in questions:
  print("Query:", query)
  print("Answer:", qa_chain.run(query))
  print()  # Add a blank line between questions for readability