from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

print("Loading DeepSeek model for conversation...")

model_id = "deepseek-ai/deepseek-llm-7b-chat"  # Better for general chat

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True,  # More aggressive quantization for 7B model
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
)

llm = HuggingFacePipeline(pipeline=pipe)

# Custom prompt template for conversation
template = """You are a helpful crypto expert assistant. Answer questions clearly and concisely.

Current conversation:
{history}
"""
