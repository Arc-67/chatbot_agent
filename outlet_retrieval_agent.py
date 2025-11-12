import json
from typing import List, Dict, Any
from sqlalchemy import select, or_, func
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Import the DB session and a helper to create it
from outlet_model import SessionLocal, Outlet

# --- 1. Define the Safe SQL Tools ---
# These tools use the SQLAlchemy ORM, preventing any SQL injection.

# --- NEW: Define a safe limit for results ---
MAX_RESULTS = 10

def get_db():
    """Helper function to get a new DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def format_outlet_results(outlets: List[Outlet], total_count: int) -> str:
    """Helper to format a list of outlets into a clean string."""
    if not outlets:
        return "No outlets found matching that criteria."
    
    results = []
    for outlet in outlets:
        services = json.loads(outlet.services) # Convert JSON string back to list
        results.append(
            f"Name: {outlet.name}\n"
            f"Address: {outlet.address}\n"
            f"Hours: {outlet.operating_hours}\n"
            f"Services: {', '.join(services)}"
        )
    
    # --- NEW: Add the count to the summary ---
    summary = f"Found {total_count} matching outlets. Showing the first {len(outlets)}:\n\n"
    return summary + "\n\n".join(results)

@tool
async def find_outlets_by_location(location: str) -> str:
    """
    Finds ZUS outlets in a specific location (e.g., city, state, or address keyword).
    Use this for queries like "outlets in Petaling Jaya" or "stores in KL".
    """
    db: Session = next(get_db())
    search_term = f"%{location}%"
    
    # --- Create the base query first ---
    base_stmt = select(Outlet).where(
        or_( # ilike is case-insensitive
            Outlet.address.ilike(search_term),
            Outlet.city.ilike(search_term),
            Outlet.state.ilike(search_term)
        )
    )
    
    # --- Get the total count first ---
    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total_count = db.scalar(count_stmt)
    
    if total_count == 0:
        return "No outlets found matching that criteria."

    # --- NEW: Apply limit to the main query ---
    stmt = base_stmt.limit(MAX_RESULTS)
    outlets = db.scalars(stmt).all()
    
    return format_outlet_results(outlets, total_count)

@tool
async def find_outlets_by_service(service: str) -> str:
    """
    Finds ZUS outlets that offer a specific service.
    Use this for queries like "which outlets are open 24 hours?" or "outlets with Dine-in".
    """
    db: Session = next(get_db())
    # This queries the JSON string directly.
    search_term = f'%"{service}"%'
    
    # --- MODIFIED: Create the base query first ---
    base_stmt = select(Outlet).where(
        Outlet.services.ilike(search_term)
    )
    
    # --- NEW: Get the total count first ---
    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total_count = db.scalar(count_stmt)
    
    if total_count == 0:
        return "No outlets found matching that criteria."

    # --- NEW: Apply limit to the main query ---
    stmt = base_stmt.limit(MAX_RESULTS)
    outlets = db.scalars(stmt).all()
    return format_outlet_results(outlets, total_count)

@tool
async def get_operating_hours(outlet_name: str) -> str:
    """
    Gets the operating hours for a *specific* ZUS outlet by its name.
    Use this if the user asks for the hours of a single, named location.
    """
    db: Session = next(get_db())
    search_term = f"%{outlet_name}%"
    
    stmt = select(Outlet).where(Outlet.name.ilike(search_term))
    outlet = db.scalars(stmt).first()
    
    if not outlet:
        return f"Could not find an outlet named '{outlet_name}'. Try a location search."
    
    return f"The operating hours for {outlet.name} are {outlet.operating_hours}."

@tool
async def list_all_outlets() -> str:
    """Lists all available ZUS outlets in the database."""
    db: Session = next(get_db())
    
    # --- NEW: Get the total count first ---
    count_stmt = select(func.count()).select_from(Outlet)
    total_count = db.scalar(count_stmt)
    
    if total_count == 0:
        return "No outlets found in the database."

    # --- NEW: Apply limit to the main query ---
    stmt = select(Outlet).limit(MAX_RESULTS)
    outlets = db.scalars(stmt).all()
    return format_outlet_results(outlets, total_count)


# --- 2. Create the Text-to-SQL Agent ---

# List of all tools the agent can use
tools = [
    find_outlets_by_location,
    find_outlets_by_service,
    get_operating_hours,
    list_all_outlets
]

# System prompt to train the agent
system_prompt = """
You are a helpful assistant for ZUS Coffee.
Your job is to answer questions about ZUS outlets using *only* the provided tools.
You must use the tools to find information. Do not make up answers.

- For general location queries (e.g., "stores in PJ", "Ampang"), use `find_outlets_by_location`.
- For service queries (e.g., "24 hour stores", "dine-in"), use `find_outlets_by_service`.
- For hours of a *specific* store, use `get_operating_hours`.
- If the user asks for all outlets, use `list_all_outlets`.

---
IMPORTANT: The tools will only return a maximum of 10 results at a time, 
even if more matches are found. You should clearly state this, 
for example: "I found 85 matching outlets. Here are the first 10:"
---

Once you have the information from the tool, present it clearly to the user as your final answer.
"""

# Create the LLM, prompt, and agent
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

# Create the AgentExecutor
# This is the object we will call from our FastAPI endpoint
sql_agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False,
    max_iterations=5,
    # It allows the agent to return the tool's output directly to the user as the final answer.
    return_intermediate_steps=False 
)