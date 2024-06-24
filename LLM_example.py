from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
import torch
import time

def load_llm_model():
    start_time = time.time()

    model_name = "Breeze-7B-32k-Instruct-v1_0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model `{model_name}`...")

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
    print("Model loading time: %.2fs\n" % (time.time() - start_time))

    return tokenizer, model

tokenizer, model = load_llm_model()
while True:

    
    prompt = input("\nEnter your prompt: ")

    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        streamer=TextStreamer(tokenizer=tokenizer),
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75,
        repetition_penalty=1.1,
    )
    outputs = tokenizer.batch_decode(generated_ids)
    # print(outputs[0]) # re-print the whole output text
    print("Inference time: %.2fs" % (time.time() - start_time))
