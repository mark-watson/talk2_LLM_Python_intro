from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini") # context window of 128k tokens

text = open('economics.txt', 'r').read()

questions = [
    "Who says economics is bullshit?",
    "Who started the Austrian School of Economics?",
]

for query in questions:
    print("Query:", query)
    prompt = f"Answer this question:\n\n{query}\n\ngiven this text:\n\n{text}\n\nAnswer:"

    messages = [
      ("system", "You answer a question, given a large block of text.", ),
      ("human", prompt,),
    ]
    answer = llm.invoke(messages)
    print("Answer:", answer.content)
