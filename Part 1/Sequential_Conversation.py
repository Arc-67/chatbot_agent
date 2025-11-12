import os
from dotenv import load_dotenv
from typing import Any
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# ------------------------ Imports & API Keys ----------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# below should not be changed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# you can change this as preferred
os.environ["LANGCHAIN_PROJECT"] = "chatbot_agent"

# ------------------------ ConversationSummaryBufferMemory ----------------------------------------------------
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

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, 
        keep only the last 'k' messages and 
        generate new summary by combining information from dropped messages.
        """
        existing_summary: SystemMessage | None = None
        old_messages: list[BaseMessage] | None = None

        # check if there is already a summary message
        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
            print(">> Found existing summary")
            existing_summary = self.messages.pop(0) # remove old summary from messages

        # add the new messages to the history
        self.messages.extend(messages)

        # check if there is too many messages
        if len(self.messages) > self.k:
            print(
                f">> Found {len(self.messages)} messages, dropping "
                f"oldest {len(self.messages) - self.k} messages.")
            
            # pull out the oldest messages
            num_to_drop = len(self.messages) - self.k
            old_messages = self.messages[:num_to_drop] # self.messages[:self.k]

            # keep only the most recent messages
            self.messages = self.messages[-self.k:]

        # if no old_messages, no new info to update the summary
        if old_messages is None:
            print(">> No old messages to update summary with")
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

        print(f">> New summary: {new_summary.content}")
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

# ------------------------ Initialize LLM model ----------------------------------------------------
# For temperature=0 for normal accurate responses
llm = ChatOpenAI(temperature=0.0, model="gpt-4.1-nano")

# ------------------------ Prompt Template ----------------------------------------------------
system_prompt = """
You are a conversational AI assistant that uses the conversation history and tool outputs as your only sources of truth.
Guidelines:
1. Answer only using information from the chat history or tool results - never invent or assume facts.
2. If key information is missing, ask one short clarifying question instead of guessing
3. If the user changes topic or interrupts, handle it naturally - answer the new query, but retain memory of previous context for when they return
4. Be concise, factual, and context-aware. Avoid repetition or over-explanation
5. When resuming after an interruption, reuse remembered context if its relevant
Your goal is to respond clearly and naturally while maintaining accurate continuity across turns.
"""
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

# ------------------------ initialize memory ----------------------------------------------------
chat_map = {}

# ------------------------ Chatbot Chain ---------------------------------------------------- 
pipeline = prompt_template | llm
pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key = "query",
    history_messages_key= "chat_history",
    history_factory_config= [
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="llm",
            annotation=ChatOpenAI,
            name="LLM",
            description="The LLM to use for the conversation summary",
            default=llm,
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=6,
        )
    ]
)

# Prints Conversation History
def print_history(session_id: str):
    print("\n=== Conversation history ===")
    history = chat_map.get(session_id)

    if history is None:
        print("(no history)")
        return
    
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            role_label = "Human"
        elif isinstance(msg, AIMessage):
            role_label = "AI"
        elif isinstance(msg, SystemMessage):
            role_label = "System (Summary)"
        else:
            role_label = msg.__class__.__name__  # fallback to type name

        content = getattr(msg, "content", str(msg))
        print(f"\n{role_label}: {content}")
    
    print("============================\n")



def happy_path(session_id: str = "session_id", k: int = 6):
    # ensure a fresh session
    if session_id in chat_map:
        chat_map[session_id].clear()
    else:
        # ensure memory object exists
        get_chat_history(session_id, llm=llm, k=k)

    print("\n ******** Happy Path ********")

    for i, msg in enumerate([
        "Hi, my name is James Potter",
        "I'm looking for a starbucks branch.",
        "The branch I am currently looking at is in Subang Jaya.",
        "This branch opens at 8am",
        "What is my name again?"
    ]):
        print(f"---\nMessage {i+1}\n---\n")
        pipeline_with_history.invoke(
            {"query": msg},
            config={"session_id": session_id, "llm": llm, "k": k}
        )

    print_history(session_id = session_id)


def idk_path(session_id: str = "session_id", k: int = 6):
    # ensure a fresh session
    if session_id in chat_map:
        chat_map[session_id].clear()
    else:
        # ensure memory object exists
        get_chat_history(session_id, llm=llm, k=k)

    print("\n ******** IDK Path ********")

    for i, msg in enumerate([
        "Hi, my name is James Potter",
        "I'm looking for a starbucks branch.",
        "By the way, is there an outlet in Subang Jaya?",
        "What's the opening time for the outlet in Subang Jaya?"
    ]):
        print(f"---\nMessage {i+1}\n---\n")
        pipeline_with_history.invoke(
            {"query": msg},
            config={"session_id": session_id, "llm": llm, "k": k}
        )

    print_history(session_id = session_id)


def interrupted_path(session_id: str = "session_id", k: int = 6):
    # ensure a fresh session
    if session_id in chat_map:
        chat_map[session_id].clear()
    else:
        # ensure memory object exists
        get_chat_history(session_id, llm=llm, k=k)

    print("\n ******** Interrupted Path ********")

    for i, msg in enumerate([
        "Hi, my name is James Potter",
        "I'm looking for a starbucks outlet.",
        "Also, do you ship internationally?",    # interruption (assistant should answer)
        "I'm looking for a outlet in Subang Jaya",
        "What is my name again?"
    ]):
        print(f"---\nMessage {i+1}\n---\n")
        pipeline_with_history.invoke(
            {"query": msg},
            config={"session_id": session_id, "llm": llm, "k": k}
        )

    print_history(session_id = session_id)



if __name__ == '__main__':
    # Output Happy Path test results
    happy_path(session_id = "id_happy", k = 6)

    # Output IDK Path test results
    idk_path(session_id = "id_idk", k = 6)

    # Output Interrupted Path test results
    interrupted_path(session_id = "id_interrupted", k = 6)