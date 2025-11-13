from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --------- Initialize OpenAI embedding and LLM ----------
# Set the K-value for "top-k"
TOP_K = 3

# Initialize Embeddings (used to connect to Pinecone)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the LLM (for summarizing)
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# ---------- load vector store index object ----------
# Set your index name
INDEX_NAME = "mind-hive-test" 

# Connect to your existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_key="text"
)

# ---------- Setup custom prompt ----------
# Define the custom prompt for summarizing
custom_prompt_template = """
You are an AI assistant for a drinkware product catalog.
Your primary goal is to provide a summary of the products found in the retrieved context.

Guidelines:
1.  Summarize What You Find: Use the user's question to identify relevant products in the context. Summarize the products you find.
2.  Be Factual: Rely *solely* on the provided context. Do not invent products, features, or any details not present in the context.
3.  Handle Partial Matches (Most Important): If the user asks for "blue mugs" and the context only contains "red mugs," *summarize the red mugs.* It is more helpful to show the user what was found.
4.  Handle No Context: If the query is out of context, politely state that you can only provide information about drinkware products.

Context: {context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

# Build the final RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": TOP_K}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True  # This is useful to see what was retrieved
)