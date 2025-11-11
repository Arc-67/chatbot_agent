import asyncio
from typing import Any, Union, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import BaseMessage

from custom_agent import QueueCallbackHandler, agent_executor, get_chat_history, llm_memory

# initilizing our application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# streaming function
async def token_generator(
        content: str, 
        streamer: QueueCallbackHandler, 
        chat_memory: Optional[Union[list[BaseMessage], object]] = None, 
        session_id: str = "session_id_00"):
    
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True,  # set to True to see verbose output in console
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

# invoke function
@app.post("/invoke")
async def invoke(content: str, session_id: str = "test_1"):
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