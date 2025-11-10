from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load DeepSeek model
model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"

print("Loading DeepSeek model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
)

llm = HuggingFacePipeline(pipeline=pipe)

# DeepSeek-specific prompt template
# DeepSeek uses a specific format for instructions
deepseek_template = """### Instruction:
{instruction}

### Response:
"""

prompt = PromptTemplate(
    template=deepseek_template,
    input_variables=["instruction"]
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Test with different prompts
test_cases = [
    "Explain what is a REST API in simple terms.",
    "Write a Go function to reverse a string.",
    "What's the difference between TCP and UDP?",
]

print("\n" + "="*60)
for i, instruction in enumerate(test_cases, 1):
    print(f"\nTest {i}: {instruction}")
    print("-"*60)
    result = chain.invoke({"instruction": instruction})
    print(result['text'])
    print("="*60)