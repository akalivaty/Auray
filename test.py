import time

import torch
import transformers
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

model_name = "Breeze-7B-32k-Instruct-v1_0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True,  # try to limit RAM
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),  # load model in low precision to save memory
    # attn_implementation="flash_attention_2",
)
# Building a LLM QNA chain
text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=300,
    streamer=streamer,
)

llama_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Read Markdown File

start_time = time.time()

# Loading and splitting the document
loader = UnstructuredMarkdownLoader("CNS16190-zh_TW.md")
data = loader.load()
print(data)

print("\ntime: %.2fs\n" % (time.time() - start_time))

# MarkdownTextSplitter & RecursiveCharacterTextSplitter

start_time = time.time()

# Chunk text
md_splits = MarkdownTextSplitter().split_documents(data)
print(f"md_splits len: {len(md_splits)}")
# print(md_splits[0])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
chunked_documents = text_splitter.split_documents(md_splits)
print(f"chunked_documents len: {len(chunked_documents)}")
for chunk in chunked_documents:
    print(chunk)

print("\ntime: %.2fs\n" % (time.time() - start_time))

# Build Retriever

start_time = time.time()

# Load chunked documents into the Qdrant index
db = Qdrant.from_documents(
    chunked_documents,
    HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1"),
    location=":memory:",
)

retriever = db.as_retriever()
retrieval_chain = RetrievalQA.from_llm(llm=llama_llm, retriever=retriever)

print("\ntime: %.2fs\n" % (time.time() - start_time))

while True:
    # 用中文列舉所有cns16190的適用範圍
    query = input("enter query: ")

    start_time = time.time()
    response = retrieval_chain.invoke(query)

    # Extract only the necessary part from the response
    start_idx = response.find("Helpful Answer: ")
    if start_idx != -1:
        response = response[start_idx + len("Helpful Answer: ") :]

    print(response)
    print("\nInference time: %.2fs\n" % (time.time() - start_time))
