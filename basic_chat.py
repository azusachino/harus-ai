from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# DeepSeek-Coder model (great for coding tasks)
# model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
model_id = "deepseek-ai/deepseek-llm-7b-chat"
# Alternative: "deepseek-ai/deepseek-llm-7b-chat" for general chat

print(f"Loading {model_id}...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto",  # Automatically use GPU if available
    trust_remote_code=True,
    load_in_8bit=True  # 8-bit quantization to save memory
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1
)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Test it
print("\n" + "="*50)
print("Testing DeepSeek model...")
print("="*50 + "\n")

response = llm.invoke("how did bad meme influences new generation.")
print(response)
