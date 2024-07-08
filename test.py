import dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()

with open("docs/CNS16190-zh_TW_only_provision.md", "r", encoding="utf-8") as f:
    data = f.read()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

md_splits = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
).split_text(data)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=10,
    separators=[
        "\n\n",
        "\n",
        " ",
        "。",
        "，",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
)
chunked_documents = text_splitter.split_documents(md_splits)

embeddings = OpenAIEmbeddings()
reviews_vector_db = Qdrant.from_documents(
    chunked_documents,
    OpenAIEmbeddings(),
    location=":memory:",
)
retriever = reviews_vector_db.as_retriever()
chat_model = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "根據以下內容回答:\n\n{context}"),
        ("user", "問題: {input}"),
    ]
)
document_chain = create_stuff_documents_chain(chat_model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input(">>> ")
while input_text.lower() != "bye":
    response = retrieval_chain.invoke({"input": input_text, "context": context})
    print(response["answer"])
    context = response["context"]
    input_text = input(">>> ")
