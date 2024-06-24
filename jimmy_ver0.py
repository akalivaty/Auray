import os
import uuid

import torch
import transformers
from flask import Flask, jsonify, request
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
ALLOWED_EXTENSIONS = {"pdf"}

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Load the Llama-2 Model
model_name = "Breeze-7B-32k-Instruct-v1_0"
model_config = transformers.AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load pre-trained config
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# Building a LLM QNA chain
text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=300,
)

llama_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
file_id = None
retrieval_chain = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_file():
    global file_id
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file_id)
        file.save(filepath)

        # Placeholder for your PDF processing logic
        process_pdf(filepath)

        return "File uploaded & processed successfully. You can begin querying now", 200


def process_pdf(filepath):
    global retrieval_chain
    # Loading and splitting the document
    loader = PyPDFLoader(filepath)
    docs = loader.load_and_split()
    # Chunk text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(docs)

    # Load chunked documents into the Qdrant index
    db = Qdrant.from_documents(
        chunked_documents,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
        location=":memory:",
    )
    retriever = db.as_retriever()
    retrieval_chain = RetrievalQA.from_llm(llm=llama_llm, retriever=retriever)


@app.route("/query", methods=["POST"])
def query():
    global retrieval_chain
    data = request.get_json()  # Ensure the request payload is parsed as JSON
    if not data or "query" not in data:
        return "Query not provided", 400

    query = data["query"]
    response = retrieval_chain.run(query)

    # Extract only the necessary part from the response
    start_idx = response.find("Helpful Answer: ")
    if start_idx != -1:
        response = response[start_idx + len("Helpful Answer: ") :]

    # Decode the response to UTF-8
    # response = response.encode('utf-8')

    return jsonify({"response": response}), 200


if __name__ == "__main__":
    app.run(debug=True)

'''
curl -X POST -F "file=@E:\dev\auray\CNS16190-zh_TW.pdf" http://localhost:5000/upload

$response = Invoke-RestMethod -Uri 'http://localhost:5000/query' -Method Post -ContentType 'application/json' -Body '{"query":"用中文總結cns16190第五章的內容"}'
$response.response
'''