import os

import dotenv
import torch
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
    pipeline,
)

dotenv.load_dotenv()
access_token = os.getenv("hf_access_token")

# Build HF LLM

# model_name = "MediaTek-Research/Breeze-7B-32k-Instruct-v1_0"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir="llm_model/", trust_remote_code=True, token=access_token
)
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="llm_model/",
    token=access_token,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
)

pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=True,
    max_new_tokens=2048,
    streamer=streamer,
)

hfPipeline = HuggingFacePipeline(pipeline=pipe)

# Build HF embeddings

# https://huggingface.co/spaces/mteb/leaderboard
sentence_transformer_model = "lier007/xiaobu-embedding-v2"  # rank 1 in chinese
# sentence_transformer_model = "iampanda/zpoint_large_embedding_zh" # rank 4 in chinese
# sentence_transformer_model = "dunzhang/stella_en_400M_v5" # rank 6 in english
# sentence_transformer_model = "Alibaba-NLP/gte-large-en-v1.5" # rank 21 in english

hf_embeddings_model = HuggingFaceEmbeddings(
    model_name=sentence_transformer_model,
    cache_folder="sentence_transformer_model",
    model_kwargs={"trust_remote_code": True},
)

# Build HF vectorstore

document_root_path = "docs"
documents = [
    "CNS16190-zh_TW.md",
    "CNS16190-zh_TW_only_provision.md",
    "ts_103701_split.pdf",
]
document_idx = 1

embedding_cache_path = "embedding_cache"
db_collection_names = [
    "CNS16190_md_hf_xiao_emb_1000_200",
    "TS103701_pdf_op_emb_1000_200",
    "TS103701_pdf_hf_stella_emb_1000_200",
]
db_collection_idx = 0

mode = "md"

if mode == "md":
    if os.path.isdir(
        os.path.join(
            embedding_cache_path, "collection", db_collection_names[db_collection_idx]
        )
    ):
        # database already exists, load it
        vectorstore = Qdrant.from_existing_collection(
            embedding=hf_embeddings_model,
            path=embedding_cache_path,
            collection_name=db_collection_names[db_collection_idx],
        )
    else:
        # database does not exist, create it
        loader = UnstructuredMarkdownLoader(
            os.path.join(document_root_path, documents[document_idx]), mode="elements"
        )
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(doc)

        vectorstore = Qdrant.from_documents(
            splits,
            embedding=hf_embeddings_model,
            path=embedding_cache_path,
            collection_name=db_collection_names[db_collection_idx],
        )

if mode == "pdf":
    if os.path.isdir(
        os.path.join(
            embedding_cache_path, "collection", db_collection_names[db_collection_idx]
        )
    ):
        # database already exists, load it
        vectorstore = Qdrant.from_existing_collection(
            embedding=hf_embeddings_model,
            path=embedding_cache_path,
            collection_name=db_collection_names[db_collection_idx],
        )
    else:
        pdf_loader = PyPDFLoader(
            os.path.join(document_root_path, documents[document_idx])
        )
        pdf_doc = pdf_loader.load()
        pdf_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        pdf_splits = pdf_text_splitter.split_documents(documents=pdf_doc)

        vectorstore = Qdrant.from_documents(
            pdf_splits,
            embedding=hf_embeddings_model,
            path=embedding_cache_path,
            collection_name=db_collection_names[db_collection_idx],
        )

retriever = vectorstore.as_retriever()


# Ask LLM

prompt = hub.pull("rlm/rag-prompt")

prompt.messages[
    0
].prompt.template = "你是一個資安專家，使用以下檢索到的背景資料回答問題，如果不知道答案就說不知道。\n背景資料：{context} \n問題：{question} \n答案："

ls_run_name = "Provisino_Search_HF"
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | prompt | hfPipeline
).with_config({"run_name": ls_run_name})

questions = "「若使用預先安裝之每裝置唯一通行碼，則應以機制產生此等通行碼，以降低對某類別或型式裝置的自動化攻擊之風險」符合哪一項控制措施？" # 控制措施5.1-2

rag_chain.invoke(questions)
