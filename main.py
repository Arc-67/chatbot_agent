import os
from dotenv import load_dotenv
# ------------------------ API Keys -------------------------------------------
# Load environment variables from .env file
load_dotenv()
# ----------------------------------------------------------------------------

import asyncio
from typing import Any, Union, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, status, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import BaseMessage

from custom_agent import QueueCallbackHandler, agent_executor, get_chat_history, llm_memory, clear_all_history
# --- Imports for the product endpoint ---
from product_retrieval import qa_chain

# --- Imports for the outlets endpoint ---
from outlet_retrieval_agent import sql_agent_executor # <-- Import the new SQL agent
from outlet_model import create_db_and_tables, populate_database, is_database_empty # <-- Import DB functions

# --- Define the background clearer chat history memory task ---
async def background_history_clearer():
    """
    This task runs in the background for the entire lifespan of the server.
    It wakes up every hour and calls the function to clear the in-memory chat map.
    """
    while True:
        await asyncio.sleep(3600)  # Sleep for 1 hour (3600 seconds)
        clear_all_history()
    
# --- Database Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs when the FastAPI app starts.
    It creates the database and populates it *only if it's empty*.
    """
    print("FastAPI startup: Initializing database...")
    # This is always safe to run, it just checks if tables exist
    create_db_and_tables() 
    
    # --- MODIFIED LOGIC ---
    if is_database_empty():
        print("Database is empty. Populating from 'outlets.jsonl'...")
        populate_database()
        print("Database population complete.")
    else:
        print("Database is already populated.")
    
    print("FastAPI startup: Database setup complete.")

    # clears chat history periodically
    asyncio.create_task(background_history_clearer())
    print("FastAPI startup: Launched background task to clear chat history every 1h.")
    
    # This 'yield' is the point where the app is running
    yield
    
    # Code after the yield would run on shutdown
    print("FastAPI shutdown complete.")

# ------------------------ FastAPI ----------------------------------------------------
# initilizing our application
app = FastAPI(lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",      # Original origin (maybe for a React app)
#         "http://127.0.0.1:5500"       # Your new HTML file's origin
#     ],  # Your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# streaming function
async def token_generator(
        content: str, 
        streamer: QueueCallbackHandler, 
        chat_memory: Optional[Union[list[BaseMessage], object]] = None, 
        session_id: str = "session_id_00"):
    
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True,
        chat_memory = chat_memory,
        session_id = session_id
    ))
    
    # initialize various components to stream
    async for token in streamer:
        try:
            if token == "<<STEP_END>>":
                # send end of step token
                # print("</step>", flush=True)
                yield "</step>"

            elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
                if tool_name := tool_calls[0]["function"]["name"]:
                    # send start of step token followed by step name tokens
                    # print(f"<step><step_name>{tool_name}</step_name>", flush=True)
                    yield f"<step><step_name>{tool_name}</step_name>"

                if tool_args := tool_calls[0]["function"]["arguments"]:
                    # tool args are streamed directly, ensure it's properly encoded
                    # print(f"{tool_args}", end="", flush=True)
                    yield tool_args
                    
                # print(f"\n{tool_calls[0]}", end="", flush=True)
                    
        except Exception as e:
            print(f"Error streaming token: {e}")
            continue

    final_answer_call = await task
    # print("\n")
    # print(final_answer_call)
    # print("\n")
    
    # if final_answer_call.get("args"):
    #     print(f"Bot Output: {final_answer_call["args"]["answer"]}")
    #     print(f"Tools Used: {final_answer_call["args"]["tools_used"]}")

# invoke AT Agent function
@app.post("/invoke", status_code=status.HTTP_202_ACCEPTED) #
async def invoke(content: str = Query(...),
                 session_id: str = Query("test_1"),
                 test_error: bool = Query(False, description="Set to true to simulate a 500 error")):
    """
    Invoves LLM Agent with tools:
    1. calculator tools
    2. /product enpoint as a tool
    3.
    """
    if test_error:
        print("--- SIMULATING HTTP 500 ERROR (/invoke) ---")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Simulated Internal Server Error for /invoke.")

    queue: asyncio.Queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)

    # chat history object
    k = 6
    chat_memory = get_chat_history(session_id=session_id, llm=llm_memory, k=k)

    # return the streaming response
    return StreamingResponse(
        token_generator(content, streamer, chat_memory=chat_memory, session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# products endpoint
@app.get("/products", status_code=status.HTTP_200_OK) #
async def get_product_summary(query: str = Query(..., min_length=3, description="Query for product summary"),
                              test_error: bool = Query(False, description="Set to true to simulate a 500 error")):
    """
    Retrieves top-k product documents and returns an AI-generated summary.
    This is a non-streaming, simple JSON endpoint.
    """
    
    if test_error:
        print("--- SIMULATING HTTP 500 ERROR (/products) ---")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Simulated Internal Server Error for /products.")

    if not qa_chain:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Product RAG chain is not available.")

    try:
        # Use 'ainvoke' for an asynchronous call
        result = await qa_chain.ainvoke({"query": query})
        
        # Return the summary and the source documents
        return {
            "query": query,
            "summary": result['result'],
            "source_documents": result['source_documents']
        }
    except Exception as e:
        # Handle any errors that occur during the RAG chain
        return {"error": f"An error occurred: {str(e)}"}
    

@app.get("/outlets")
async def get_outlet_info(
    query: str = Query(..., min_length=3, description="Natural language query for ZUS outlets"),
    test_error: bool = Query(False, description="Set to true to simulate a 500 error")):
    """
    Translates a natural language query to SQL, executes it against
    the ZUS outlet database, and returns the results.
    """
    print(f"--- Received /outlets query: {query} ---")

    # --- THIS IS THE NEW TEST CODE ---
    if test_error:
        print("--- SIMULATING HTTP 500 ERROR ---")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Simulated Internal Server Error for testing.")
    # --- END OF TEST CODE ---

    try:
        # Use 'ainvoke' to call the agent asynchronously
        # The agent will handle figuring out which tool to call.
        result = await sql_agent_executor.ainvoke({
            "input": query,
            "chat_history": [] # Pass an empty history for this single-turn endpoint
        })
        
        # The final answer is in the 'output' key
        return {
            "query": query,
            "result": result['output']
        }
    except Exception as e:
        print(f"Error in /outlets endpoint: {e}")
        return {"error": f"An error occurred: {str(e)}"}
    
@app.get("/")
async def get_index():
    """
    Serves the main chat application HTML file.
    """
    # Assumes 'ai_chatbot.html' is in the same directory as 'main.py'
    return FileResponse("ai_chatbot.html", media_type="text/html")