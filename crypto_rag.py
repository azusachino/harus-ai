from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Sample crypto documents (replace with your actual research)
crypto_docs = [
    """Bitcoin is a decentralized cryptocurrency that uses Proof of Work (PoW) 
    consensus mechanism. It was created by Satoshi Nakamoto in 2009. Bitcoin's 
    block time is approximately 10 minutes, and its maximum supply is capped at 
    21 million coins.""",
    
    """Ethereum is a blockchain platform that supports smart contracts. It 
    transitioned from Proof of Work to Proof of Stake (PoS) in September 2022 
    through an upgrade called "The Merge". Ethereum's block time is around 12 
    seconds.""",
    
    """DeFi (Decentralized Finance) refers to financial applications built on 
    blockchain technology. Common DeFi protocols include Uniswap (DEX), Aave 
    (lending), and MakerDAO (stablecoin). Total Value Locked (TVL) is a key 
    metric for measuring DeFi adoption.""",
    
    """Layer 2 solutions like Arbitrum and Optimism help scale Ethereum by 
    processing transactions off the main chain. They use rollup technology to 
    bundle multiple transactions together, reducing fees and increasing throughput."""
]

print("Setting up RAG system with DeepSeek...")

# 1. Prepare documents
documents = [Document(page_content=doc) for doc in crypto_docs]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# 2. Create embeddings
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# 3. Create vector store
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. Load DeepSeek model
print("Loading DeepSeek model...")
model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"

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
    max_new_tokens=300,
    temperature=0.3,  # Lower temperature for factual answers
    top_p=0.9,
)

llm = HuggingFacePipeline(pipeline=pipe)

# 5. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# 6. Query the knowledge base
print("\n" + "="*70)
print("RAG System Ready! Asking questions...")
print("="*70 + "\n")

questions = [
    "What consensus mechanism does Bitcoin use?",
    "When did Ethereum switch to Proof of Stake?",
    "What are Layer 2 solutions?",
    "What is DeFi and give examples of protocols?"
]

for question in questions:
    print(f"\nQuestion: {question}")
    print("-"*70)
    
    result = qa_chain.invoke({"query": question})
    print(f"Answer: {result['result']}")
    
    print("\nSource documents:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"{i}. {doc.page_content[:100]}...")
    print("="*70)