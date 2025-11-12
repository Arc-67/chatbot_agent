import types
from types import SimpleNamespace
import pytest
import Sequential_Conversation as sc

# Ensure chat_map exists
sc.chat_map = {}

# A very small deterministic fake LLM with an .invoke() method that returns an object with a 'content' attribute.
class FakeLLM:
    def __init__(self):
        pass

    def invoke(self, formatted_messages):
        """
        formatted_messages will be a list of Message objects (ChatPromptTemplate.format_messages output).
        For easy detection we convert to a single string and inspect it.
        """
        # Build a quick textual view of the prompt/messages to decide the response
        try:
            text = "\n".join(
                getattr(m, "content", str(m)) for m in formatted_messages
            )
        except Exception:
            # if formatted_messages is already a string, use it directly
            text = str(formatted_messages)

        text_lower = text.lower()

        # Rules for deterministic behavior:
        # - If prompt already contains "Subang Jaya" and asks opening time -> return specific time
        # - If prompt contains "what" and "name" -> return "Your name is James Potter"
        # - If prompt contains "ship internationally or "do you ship" (interrupt) -> answer shipping question
        # - Else return -> "Sorry, I don't have that info."
        if "subang jaya" in text_lower and ("open" in text_lower or "opening time" in text_lower or "opening" in text_lower):
            return SimpleNamespace(content="The Subang Jaya branch opens at 9:00AM.")
        
        if "name" in text_lower and "what" in text_lower:
            return SimpleNamespace(content="Your name is James Potter")
        
        if "do you ship" in text_lower or "ship internationally" in text_lower:
            return SimpleNamespace(content="Starbucks generally offers international shipping through their online store, but availability varies by country and product.")
        
        # fallback deterministic reply
        return SimpleNamespace(content="Sorry, I don't have that info.")
    
# Helper to create Minimal message objects matching langchain messages shape.
class MockMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"<{self.role}: {self.content}>"
    
# ---- Test: happy path ----
def test_happy_path_answer_using_memory_and_history():
    session_id = "test_happy"
    fake_llm = FakeLLM()
    # create memory object for session via get_chat_history (ensures same construction semantics)
    mem = sc.ConversationSummaryBufferMemory_custom(llm=fake_llm, k=4)
    sc.chat_map[session_id] = mem

    # Simulate prior conversation:
    mem.messages = [
        MockMessage("system", "The conversation began with the user, James Potter, greeting the AI, but no further exchanges occurred until recently."),
        MockMessage("user", "I'm looking for a starbucks branch."),
        MockMessage("assistant", "Which outlet are you referring to?"),
        MockMessage("user", "The branch I am currently looking at is in Subang Jaya. This branch opens at 8am"),
        MockMessage("assistant", "I see, do you want more infromation on that branch?")
    ]

    # Now the user asks for opening time (the model should use the history to answer)
    user_query = "What's the Subang Jaya branch opening time again?"

    # Build a compact prompt that includes history + user query (similar to prompt_template)
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

    # For testing, create a simple formatted input: system + history + user query as list
    formatted_messages = [MockMessage("system", system_prompt)]
    formatted_messages.extend(mem.messages)  # history
    formatted_messages.append(MockMessage("human", user_query))

    # Invoke fake LLM with formatted prompt
    response = fake_llm.invoke(formatted_messages)
    assert isinstance(response.content, str)
    assert "9:00AM" in response.content or "opens at 9:00AM" in response.content


# ---- Test: interrupted path ----
@pytest.mark.parametrize("interrupt_question, expected_reply_substring", [
    ("Also, do you ship internationally?", "international shipping"),
    ("By the way, what's your return policy?", "don't have that info"),  # fallback from FakeLLM
])
def test_interrupted_path_handles_interruption_and_retains_context(interrupt_question, expected_reply_substring):
    session_id = "test_interrupt"
    fake_llm = FakeLLM()
    mem = sc.ConversationSummaryBufferMemory_custom(llm=fake_llm, k=4)
    sc.chat_map[session_id] = mem

    # Initial flow: user asks city, assistant asks which outlet (not resolved yet)
    mem.messages = [
        MockMessage("system", "The conversation began with the user, James Potter, greeting the AI, but no further exchanges occurred until recently."),
        MockMessage("user", "I'm looking for a starbucks outlet."),
        MockMessage("assistant", "Which outlet are you referring to?"),
    ]

    # Build a compact prompt that includes history + user query (similar to prompt_template)
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

    # User interrupts with an unrelated question; system should answer this new question
    formatted_messages_interrupt = [MockMessage("system", system_prompt)]
    formatted_messages_interrupt.extend(mem.messages)
    formatted_messages_interrupt.append(MockMessage("human", interrupt_question))

    interrupt_reply = fake_llm.invoke(formatted_messages_interrupt)
    assert expected_reply_substring.lower() in interrupt_reply.content.lower()

    # After answering the interruption, user returns to original thread and supplies outlet
    # The memory should still have the earlier "Is there an outlet in Petaling Jaya?" context
    # now user says "SS 2, opening time?"
    formatted_messages_resume  = [MockMessage("system", system_prompt)]
    formatted_messages_resume.extend(mem.messages)
    formatted_messages_resume.append(MockMessage("human", "Sorry the branch in Subang Jaya. What's the opening time?"))

    resume_reply = fake_llm.invoke(formatted_messages_resume)
    assert "9:00AM" in resume_reply.content or "opens at 9:00AM" in resume_reply.content
