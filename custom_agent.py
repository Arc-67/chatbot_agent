import os
from datetime import datetime, timezone

import asyncio
from typing import Any, Union, Optional
from pydantic import BaseModel, Field
import httpx

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.base import AsyncCallbackHandler

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI

# ------------------------ LLM & Prompt ----------------------------------------------------
# LLM and Prompt Setup
llm_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.0,
    streaming=True,
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)

system_prompt = """
You are a conversational AI assistant that relies solely on the conversation history and tool outputs as your sources of truth.
Always use tools to answer the user's current question not previous questions before responding.

You have two types of tools:
1. Product Catalog: Use the 'query_product_catalog' tool for any questions about drinkware products (mugs, cups, bottles, etc.).
2. Outlet Catalog: Use 'query_outlet_catalog' for questions about ZUS Coffee outlet locations, hours, or services.
3. Calculators: Use tools like 'add', 'subtract', etc., for math questions.

When you have enough information (from tool results), you *must* call the 'final_answer' tool to provide the final response.

Guidelines:
1. Use only chat history or tool results and never invent or assume facts. 
2. Be concise, factual, and context-aware.
3. If information is missing, ask one short clarifying question.
4. If the user changes topic or interrupts, handle it naturally and retain relevant context.
5. For math problems, you must solve them by breaking them down into multiple, single steps using your calculator tools.

Your goal is to respond clearly by first selecting the correct tool, and finally call the 'final_answer' tool with the complete solution.
"""

# Arithmetic Order of Operations:
#     1. Parentheses: " ( " , " ) "
#     2. Exponentiation: " ^ " , " ** "
#     3. Multiplication and division: " * " , " / "
#     4. Addition and subtraction: " + " , " - "

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ------------------------ Chat Memory ----------------------------------------------------
class ConversationSummaryBufferMemory_custom(BaseChatMessageHistory, BaseModel):
    """
    Based on number of messages. Where if number of messages is more than k, 
    pop oldest messages and create a new summary by adding information from poped messages.
    """
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: Any = None
    k: int = Field(default_factory=int)

    def __init__(self, llm: Any, k: int):
        super().__init__(llm=llm, k=k)
        # print(f"Initializing ConversationSummaryBufferMemory_custom with k={k}")

    async def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, 
        keep only the last 'k' messages and 
        generate new summary by combining information from dropped messages.
        """
        existing_summary: SystemMessage | None = None
        old_messages: list[BaseMessage] | None = None

        # check if there is already a summary message
        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
            # print(">> Found existing summary")
            existing_summary = self.messages.pop(0) # remove old summary from messages

        # add the new messages to the history
        self.messages.extend(messages)

        # check if there is too many messages
        if len(self.messages) > self.k:
            # print(
            #     f">> Found {len(self.messages)} messages, dropping "
            #     f"oldest {len(self.messages) - self.k} messages.")
            
            # pull out the oldest messages
            num_to_drop = len(self.messages) - self.k
            old_messages = self.messages[:num_to_drop] # self.messages[:self.k]

            # keep only the most recent messages
            self.messages = self.messages[-self.k:]

        # if no old_messages, no new info to update the summary
        if old_messages is None:
            # print(">> No old messages to update summary with")
            return
        
        # construct the summary chat messages
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensuring to maintain "
                "as much relevant information as possible."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{old_messages}"
            )
        ])

        # format the messages and invoke the LLM
        new_summary = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary,
                old_messages=old_messages
            )
        )

        # call synchronous llm.invoke in a thread so we don't block the event loop:
        # loop = asyncio.get_running_loop()
        
        # new_summary = await loop.run_in_executor(
        #     None,  # default ThreadPoolExecutor
        #     lambda: self.llm.invoke(
        #         summary_prompt.format_messages(
        #             existing_summary=existing_summary,
        #             old_messages=old_messages
        #         )
        #     )
        # )

        # print(f">> New summary: {new_summary.content}")
        # prepend the new summary to the history
        self.messages = [SystemMessage(content=new_summary.content)] + self.messages


    def clear(self) -> None:
        """Clear the history."""
        self.messages = []

# function to get memory for specific session id
def get_chat_history(session_id: str, llm: ChatOpenAI, k: int = 4) -> ConversationSummaryBufferMemory_custom:
    # print(f"get_chat_history called with session_id={session_id} and k={k}")
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = ConversationSummaryBufferMemory_custom(llm=llm, k=k)
    # remove anything beyond the last
    return chat_map[session_id]

def clear_all_history():
    """
    Clears the in-memory chat history map.
    This is called by a background task in main.py to prevent memory leaks.
    """
    print("--- Clearing all chat histories... ---")
    chat_map.clear()
    print("--- Chat history cleared. ---")

chat_map = {}
llm_memory = ChatOpenAI(temperature=0.0, model="gpt-4.1-nano")

# ------------------------ Agent Tools ----------------------------------------------------
# Tools definition
# note: all tools as async to simplify later code
@tool
async def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

@tool
async def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

@tool
async def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

@tool
async def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x

@tool
async def divide(x: float, y: float) -> float:
    """Divide 'x' by 'y'."""
    if y > 0:
        return x / y
    else:
        return "Division by error y must be more than 0"

@tool
async def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final answer to the user. Or to ask a clarifying question."""
    return {"answer": answer, "tools_used": tools_used}

@tool
async def query_product_catalog(query: str) -> str:
    """
    Use this tool to find information about drinkware products, such as 
    mugs, cups, or bottles. You must provide a search query.
    Returns an AI-generated summary of any matching products found.
    """
    # print(f"--- Calling Product Tool with query: {query} ---")
    try:
        # This new dictionary forces the server to fail
        # test_params = {
        #     "query": query,
        #     "test_error": True 
        # }

        # Use httpx.AsyncClient for async requests
        async with httpx.AsyncClient() as client:
            response = await client.get(
                PRODUCT_API_URL,
                params={"query": query}, #  test_params
                timeout=10.0  # Add a 10-second timeout
            )
        
        # Raise an exception for bad status codes (like 404 or 500)
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Return the summary, or a fallback message
        return data.get("summary", "No summary was returned from the product catalog.")

    except httpx.HTTPStatusError as e:
        # This catches 500 errors from the server
        print(f"HTTP Status Error: {e.response.status_code}")
        return f"Error: The outlet catalog service is temporarily down (HTTP {e.response.status_code}). Please try again later."
    
    except httpx.RequestError as e:
        print(f"HTTP Request Error: {e}")
        return f"Error: Could not connect to the product catalog: {str(e)}"
    except Exception as e:
        print(f"Product Tool Error: {e}")
        return f"An unexpected error occurred while querying products: {str(e)}"

# --- NEW OUTLET CATALOG TOOL ---
@tool
async def query_outlet_catalog(query: str) -> str:
    """
    Use this tool to find information about ZUS Coffee outlet locations, 
    operating hours, or services. You must provide a search query.
    Returns an AI-generated answer about matching outlets.
    """
    print(f"--- Calling Outlet Tool with query: {query} ---")
    try:
        # This new dictionary forces the server to fail
        # test_params = {
        #     "query": query,
        #     "test_error": True 
        # }

        # Use httpx.AsyncClient for async requests
        async with httpx.AsyncClient() as client:
            response = await client.get(
                OUTLET_API_URL,
                params={"query": query}, # test_params
                timeout=10.0  # Add a 10-second timeout
            )
        
        response.raise_for_status()
        data = response.json()
        
        # Note: The /outlets endpoint returns "result"
        return data.get("result", "No result was returned from the outlet catalog.")

    except httpx.HTTPStatusError as e:
        # This catches 500 errors from the server
        print(f"HTTP Status Error: {e.response.status_code}")
        return f"Error: The outlet catalog service is temporarily down (HTTP {e.response.status_code}). Please try again later."
    
    except httpx.RequestError as e:
        print(f"HTTP Request Error: {e}")
        return f"Error: Could not connect to the outlet catalog: {str(e)}"
    
    except Exception as e:
        print(f"Outlet Tool Error: {e}")
        return f"An unexpected error occurred while querying outlets: {str(e)}"

# Get the base URL from an env var, defaulting to the local server for development
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

PRODUCT_API_URL = f"{API_BASE_URL}/products"
OUTLET_API_URL = f"{API_BASE_URL}/outlets"

print(f"--- Tools will call endpoints at: {API_BASE_URL} ---")

tools = [add, subtract, multiply, exponentiate, divide, query_product_catalog, query_outlet_catalog, final_answer]
# note when we have sync tools we use tool.func, when async we use tool.coroutine
name2tool = {tool.name: tool.coroutine for tool in tools}

# ------------------------ Streaming Callback ----------------------------------------------------
# Streaming Handler
class QueueCallbackHandler(AsyncCallbackHandler):
    """Callback handler that puts tokens into a queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False

    async def __aiter__(self): # outputs tokens
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                # this means we're done
                return
            if token_or_done:
                yield token_or_done
    
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        """Put new token in the queue."""

        #print(f"on_llm_new_token: {args}, {kwargs}")
        chunk = kwargs.get("chunk")
        # check for final_answer tool call
        if chunk and chunk.message.additional_kwargs.get("tool_calls"):
            if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                # this will allow the stream to end on the next `on_llm_end` call
                self.final_answer_seen = True
                
        self.queue.put_nowait(kwargs.get("chunk")) #store tokens into queue
    
    async def on_llm_end(self, *args, **kwargs) -> None:
        """Put DONE in the queue to signal completion."""

        #print(f"on_llm_end: {args}, {kwargs}")
        # this should only be used at the end of our agent execution, however LangChain
        # will call this at the end of every tool call, not just the final tool call
        # so we must only send the "done" signal if we have already seen the final_answer

        if self.final_answer_seen:
            self.queue.put_nowait("<<STEP_END>>")
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")

async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    tool_name = tool_call.tool_calls[0]["name"]
    tool_args = tool_call.tool_calls[0]["args"]
    tool_out = await name2tool[tool_name](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
        tool_call_id=tool_call.tool_calls[0]["id"]
    )

# ------------------------ Agent Executor ----------------------------------------------------
# Agent Executor
class CustomAgentExecutor:
    def __init__(self, max_iterations: int = 15):
        self.chat_history: list[BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm_agent.bind_tools(tools, tool_choice="required")
        )

        # In-memory trace list for inspection
        self.decision_trace: list[dict] = []

    async def invoke(self, input: str, 
                     streamer: QueueCallbackHandler, 
                     verbose: bool = False, 
                     chat_memory: Optional[Union[list[BaseMessage], object]] = None, 
                     session_id: str = "session_id_00") -> dict:
        # --- pick the chat history container to use for this invocation ---
        # support memory object (with .messages) or plain list
        if chat_memory is None:
            # fallback (existing behavior)
            chat_container = self.chat_history
            use_memory_api = False
        else:
            # detect if memory object (duck-typing)
            if hasattr(chat_memory, "messages") and hasattr(chat_memory, "add_messages"):
                chat_container = chat_memory  # memory object
                use_memory_api = True

            elif isinstance(chat_memory, list):
                chat_container = chat_memory
                use_memory_api = False
            else:
                # Unexpected type - fall back to list view if possible
                raise TypeError("chat_memory must be a list or a memory object with .messages/.add_messages")
                                
        # invoke the agent but we do this iteratively in a loop until reach a final answer
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        # streaming function
        async def stream(query: str) -> list[AIMessage]:
            # get the current messages list to pass to the agent prompt
            if use_memory_api:
                history_for_prompt = chat_container.messages
            else:
                history_for_prompt = chat_container

            configured_agent = self.agent.with_config(
                callbacks=[streamer]
            )
            # Initialize the output dictionary that will be populating with streamed output
            outputs = []
            # now begin streaming
            async for token in configured_agent.astream({
                "input": query,
                "chat_history": history_for_prompt,
                "agent_scratchpad": agent_scratchpad
            }):
                tool_calls = token.additional_kwargs.get("tool_calls")
                if tool_calls: # -> outputs = [tool1, tool2, tool3]
                    # first check if have a tool call id - this indicates a new tool
                    if tool_calls[0]["id"]:
                        outputs.append(token)
                    else:
                        outputs[-1] += token
                else:
                    pass
            return [
                AIMessage(
                    content=x.content,
                    tool_calls=x.tool_calls,
                    tool_call_id=x.tool_calls[0]["id"]
                ) for x in outputs
            ]

        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            tool_calls = await stream(query=input)
            # gather tool execution coroutines
            tool_obs = await asyncio.gather(
                *[execute_tool(tool_call) for tool_call in tool_calls]
            )
            # append tool calls and tool observations to the scratchpad in order
            id2tool_obs = {tool_call.tool_call_id: tool_obs for tool_call, tool_obs in zip(tool_calls, tool_obs)}
            for tool_call in tool_calls:
                agent_scratchpad.extend([
                    tool_call,
                    id2tool_obs[tool_call.tool_call_id]
                ])

            # ------------------ TOOL-USAGE LOGGING ------------------
            # Only log when a tool was requested this iteration
            if tool_calls:
                # Use first tool_call as representative (agent may call multiple tools)
                tc = tool_calls[0]
                tool_name = tc.tool_calls[0].get("name")
                tool_args = tc.tool_calls[0].get("args", {})
                obs = id2tool_obs.get(tc.tool_call_id)
                tool_result_summary = getattr(obs, "content", str(obs)) if obs is not None else None

                trace_entry = {
                    "session_id": session_id,
                    "turn": count + 1,
                    "query": input,
                    "tool": tool_name,
                    "tool_args": tool_args,
                    "tool_result_summary": tool_result_summary,
                    "timestamp": datetime.now(timezone.utc) #datetime.now(datetime.timezone.utc) + "Z"
                }

                # keep the trace in memory and print it (for screenshots / manual inspection)
                # self.decision_trace.append(trace_entry) # commented out to avoid exceeding memory allocation (TO DO: need to be cleared every set time)
                # pretty print the trace to console (one-line)
                # print("PLANNER TRACE:", trace_entry)
            
            count += 1
            # if the tool call is the final answer tool, then stop
            found_final_answer = False
            for tool_call in tool_calls:
                if tool_call.tool_calls[0]["name"] == "final_answer":
                    final_answer_call = tool_call.tool_calls[0]
                    final_answer = final_answer_call["args"]["answer"]
                    found_final_answer = True
                    break
            
            # Only break the loop if found a final answer
            if found_final_answer:
                break
            
        # --- write final messages back into the chat memory (or list) ---
        human_msg = HumanMessage(content=input)
        ai_msg = AIMessage(content=final_answer if final_answer else "No answer found")

        # If the agent exited due to reaching max iteration limit, ensure the streamer stops cleanly
        if not found_final_answer and count >= self.max_iterations:
            print(f"[WARN] Max iteration limit ({self.max_iterations}) reached - stopping agent loop.")
            try:
                streamer.queue.put_nowait("<<DONE>>")
            except Exception:
                pass

        if use_memory_api:
            # use memory object's API (ConversationSummaryBufferMemory_custom.add_messages)
            # Note: add_messages expects a list[BaseMessage]
            try:
                await chat_container.add_messages([human_msg, ai_msg])
                
            except Exception as e:
                # fallback: append to the messages list
                chat_container.messages.extend([human_msg, ai_msg])
        else:
            # plain list
            chat_container.extend([human_msg, ai_msg])

        # return the final answer in dict form
        return final_answer_call if final_answer else {"answer": "No answer found or iteration limit reached"}

# Initialize agent executor
agent_executor = CustomAgentExecutor() 