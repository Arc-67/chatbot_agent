# AI Multi-Agent Chat Application

This is a full-stack AI chat application featuring a multi-agent backend built with Python (FastAPI) and a real-time streaming frontend built with plain HTML, CSS, and JavaScript.

The application is built on an "agent-of-agents" architecture. A primary "orchestrator" agent (``/invoke``) fields user requests and intelligently delegates tasks to specialized sub-agents and tools, which are exposed as internal API endpoints:

1. **Product Agent (``/products``):** A Retrieval-Augmented Generation (RAG) agent that connects to a Pinecone vector database to find and summarize ZUS Coffee drinkware products.

2. **Outlet Agent (``/outlets``):** A secure Text-to-SQL agent that connects to a local SQLite database to answer questions about ZUS Coffee outlet locations.

The frontend chat interface visualizes this entire process, showing the agent's "thoughts" and tool calls in real-time as they happen. 

This is the link to the deployed demo: [Demo Link](https://chatbot-agent-cb3y.onrender.com) 
Note that due to deploying on a free tier, the application goes to sleep after 15 minutes of inactivity. So it may take a few minutes to starup again.

# Features

- **Multi-Agent Backend:** A main orchestrator agent that routes tasks to specialized RAG and Text-to-SQL agents.

- **Real-time Streaming:** The main chat interface streams the agent's actions and final answer token-by-token.

- **Agentic Visualization:** The UI renders the agent's internal tool calls (e.g., ``query_outlet_catalog``, ``add``, ``multiply``) as distinct "steps" in the chat.

- **Secure Text-to-SQL:** The ``/outlets`` agent uses a tool-based system, not raw SQL generation, to completely prevent SQL injection vulnerabilities.

- **Data Persistence:** The chat history is saved to ``localStorage``, so conversations are not lost on browser refresh.

- **Self-Healing Database:** The server automatically creates and populates its SQLite database from a JSONL file on its first run.

- **Downtime Testing:** Endpoints include a ``test_error=true`` flag to safely test and verify the agent's "unhappy path" and error handling.

# Architecture & Design Trade-offs
This project uses a specific set of architectural patterns, each with its own trade-offs. For the primary "orchestrator" agent (``/invoke``) more detail information on how the agent functions can be found in the document ``Documentation/Part 2 Agent Documentation.pdf``.

### 1. Chat History Memory (``custom_agent.py``)
The chat history for the main ``/invoke`` agent is handled by a custom class, ``ConversationSummaryBufferMemory_custom``.

- **How it Works:** This class holds the last ``k`` (e.g., 6) messages in memory for each ``session_id`` (user id). When a new message arrives and the limit is exceeded, the oldest messages (the ones being dropped typically 2 messages) are sent to an LLM to be summarized. This new summary is then prepended to the message list as a ``SystemMessage``.

- **Storage:** All active conversations are stored in a single, global dictionary (``chat_map``) in the server's memory. A background task (``background_history_clearer``) runs every hour to wipe this dictionary, preventing a long-term memory leak.

- **Trade-offs:**
    - **Pro:** This keeps the context window sent to the agent (e.g., ``gpt-4.1-nano``) small and fast. The agent only has to process the summary + the last ``k`` messages, not the entire conversation, saving on tokens and cost.

    - **Pro:** The summarization step allows the agent to "remember" details from much earlier in the conversation, which is superior to a simple "cut-off" buffer.

    - **Con (Cost/Latency):** It requires an additional LLM call every ``k`` messages to create the summary, which adds a small amount of latency to that specific turn. However, in this context the input into the LLM (messages + previous summary) to be summarized is relatively small.

    - **Con (Context Loss):** The quality of the "memory" is dependent on the summarizer model (gpt-4.1-nano). If the summarizer deems a detail "unimportant," it will be lost from the agent's memory forever.

    - **Con (Scalability):** This in-memory storage is not durable. If the server restarts, all conversations are lost. It also does not scale horizontally (i.e., in a multi-server environment), as a user's session is tied to the memory of a single server.

### 2. "Agent-of-Agents" via HTTP
The main ``/invoke`` agent does not call the product and outlet agents as internal Python functions. It calls them via their own HTTP endpoints (``/products``, ``/outlets``).

- **Trade-offs:**
    - **Pro (Modularity):** This is a clean "microservice" architecture. You can update, deploy, or even switch the model of the ``/products`` agent without ever touching the main ``/invoke`` agent.

    - **Con (Network Overhead):** This is slower than an in-process call. The agent has to make a full ``httpx`` network request even if it's just to itself on ``localhost`` or the Render URL. This adds latency compared to a direct Python function call.

### 3. LLM Model Choice 
This implementation uses the lightweight model (``gpt-4.1-mini``) for the main "orchestrator" agent, the other 2 agents uses the model (``gpt-4.1-nano``).

- **Trade-offs:**
    - **Pro (Speed/Cost):** Due to the lightweight models the application is extremely fast and cost-effective. It's perfectly sufficient for simple tasks and for demonstrating the core architecture.

    - **Con (Reasoning Power):** The system is brittle. The "specialist" agents (``invoke``,``/products`` and ``/outlets``) are underpowered for complex reasoning. For example, the agent can handle relatively simple query like calculate "5*(3+6)", but if asked to perform a more complex query like "2*(5*(3+6)/3)^3" the agent will likely fail. Where a more powerful model may be able to solve.

# Tech Stack

### Backend:

- **FastAPI:** For the main web server and API endpoints.

- **LangChain:** For building all AI agents, prompts, and tool-calling logic.

- **SQLAlchemy:** For the ORM and connection to the SQLite database.

- **SQLite:** For the ZUS Coffee outlet database.

- **Pinecone:** For the product vector database.

- **httpx:** For agent-to-agent (endpoint) communication.

- **Uvicorn:** As the ASGI server to run FastAPI.

### Frontend:

- Plain HTML, CSS, and JavaScript (no frameworks).

### Data:

- ``BeautifulSoup4``: Used in ``zus_product_crawl`` and ``zus_outlet_crawl_2`` to get outlet data.

# How to Run Locally

Follow these instructions to set up and run the entire application on your machine.

### Step 1: Clone the Repository

Clone this project to your local machine.

```
git clone https://github.com/Arc-67/chatbot_agent.git``
cd chatbot_agent
```

### Step 2: Set Up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```
# Create the virtual environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required Python packages using the provided requirements.txt file.

```
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

You must provide API keys for the AI models and Pinecone.

1. Create a new file named ``.env`` in the root of the project directory.

2. Add the following keys to the file, filling in your own values:

```
# .env
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="products"
```

Note: The ``PINECONE_INDEX_NAME`` must match the name of your index in Pinecone that contains your product data.

### Step 5: Manually Create Pinecone Index

This project expects you to create the Pinecone index manually. The ingestion script will add to an existing index, not create a new one.

1. Log in to your Pinecone account.

2. Go to **Indexes** and click **Create Index**.

3. Set the Index Name to match the ``PINECONE_INDEX_NAME`` from your ``.env`` file (e.g., ``products``).

4. Set the Dimensions to ``1536`` (this is required for OpenAI's ``text-embedding-3-small model``).

5. Choose your preferred pod type (e.g., s1 or p1).

6. Click Create Index.

### Step 6: Ingest Product Data into Pinecone

Now that your index is created and your ``products.jsonl`` file is present, run the ingestion script. This will read the JSONL file, create embeddings, and upload them to Pinecone.

```
python ingest.py
```

This will log its progress and print "[INFO] Done indexing." when complete.

### Step 7: Run the Application

You can now start the main FastAPI server. This single command starts the backend and serves the HTML frontend.

```
uvicorn main:app --reload
```

On the very first run, the server will detect if the database is empty and automatically populate its ``zus_outlets.db`` file from ``outlets.jsonl``.

**How to Use**

1. Once the server is running, open your browser and go to: http://127.0.0.1:8000

2. The chat interface will load. You can now test all of the agent's capabilities.

**Example Queries to Try:**

- **Simple Chat:** ``Hello, my name is James Potter?``

- **Calculator (Multi-Step):** ``/calc (10 + 5) * 2`` or ``(10 + 5) * 2``

- **Product RAG Agent:** ``/products what kind of drinkware do you have?`` or ``what kind of drinkware do you have?`` 

- **Text-to-SQL Agent:** ``/outlets show me outlets in Petaling Jaya`` or ``show me outlets in Petaling Jaya``

- **Reset Chat:** click the **Clear Chat** button or enter ``/reset``

# Project File Structure

```
.
├── .env                        # (You must create this) Stores API keys.
├── ai_chatbot.html             # The HTML, CSS, and JS for the frontend.
├── main.py                     # Main FastAPI server. Serves the HTML and all API endpoints.
├── custom_agent.py             # Defines the main /invoke agent, its tools, and streaming logic.
├── product_retrieval.py        # Defines the secure RAG pipeline for the /products endpoint.
├── outlet_retrieval_agent.py   # Defines the secure Text-to-SQL agent for the /outlets endpoint.
├── outlet_model.py             # Defines the SQLite database schema and data loading logic.
├── ingest.py                   # (Run once) Ingests product data into Pinecone.
├── products.jsonl              # (Provided) Product data for Pinecone.
├── outlets.jsonl               # (Provided)  Outlet data.
├── zus_outlets.db              # (Generated by main.py) The SQLite database.
└── requirements.txt            # All Python dependencies.
```
