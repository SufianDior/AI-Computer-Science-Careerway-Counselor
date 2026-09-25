from typing import Annotated, TypedDict
import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage


# =========================================================
# PROFILE DATABASE
# =========================================================

profile_connection = sqlite3.connect(
    "profile_memory.db",
    check_same_thread=False
)

profile_cursor = profile_connection.cursor()

profile_cursor.execute("""
CREATE TABLE IF NOT EXISTS student_profile (
    user_id TEXT,
    field TEXT,
    value TEXT,
    PRIMARY KEY (user_id, field)
)
""")

profile_connection.commit()

CURRENT_USER = "main_user"


# =========================================================
# TOOLS
# =========================================================

search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return search.run(query)


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def update_profile(field: str, value: str) -> str:
    """
    Save or update an important long-term fact about the student.

    Useful fields include:
    name
    major
    school
    career_goal
    interests
    skills
    transfer_goal
    graduation_goal
    experience
    certifications
    """

    allowed_fields = [
        "name",
        "major",
        "school",
        "career_goal",
        "interests",
        "skills",
        "transfer_goal",
        "graduation_goal",
        "experience",
        "certifications"
    ]

    field = field.lower().strip()

    if field not in allowed_fields:
        return f"Cannot save '{field}'. Allowed fields: {', '.join(allowed_fields)}"

    profile_cursor.execute("""
        INSERT INTO student_profile (user_id, field, value)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, field)
        DO UPDATE SET value = excluded.value
    """, (CURRENT_USER, field, value))

    profile_connection.commit()

    return f"Saved {field}: {value}"


@tool
def get_profile() -> str:
    """Retrieve the student's saved long-term profile."""

    profile_cursor.execute("""
        SELECT field, value
        FROM student_profile
        WHERE user_id = ?
    """, (CURRENT_USER,))

    rows = profile_cursor.fetchall()

    if not rows:
        return "No student profile information has been saved yet."

    profile = []

    for field, value in rows:
        profile.append(f"{field}: {value}")

    return "\n".join(profile)


tools = [
    web_search,
    get_current_time,
    update_profile,
    get_profile
]


# =========================================================
# LLM
# =========================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3
)

llm_with_tools = llm.bind_tools(tools)


# =========================================================
# STATE
# =========================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an AI Career Pathway Counselor.

Your job is to help students plan careers in:
- Computer Science
- Software Engineering
- Artificial Intelligence
- Machine Learning
- Cybersecurity
- Data Science
- and related technology fields.

You have persistent student profile memory.

IMPORTANT MEMORY RULES:

When the user tells you an important long-term fact about themselves,
use the update_profile tool to save it.

Also; IMPORTANT TOOL USAGE RULES:

When the student explicitly asks you to remember,
save, or update their personal information,
you MUST call the update_profile tool.

Never claim you saved information unless
the update_profile tool actually executed
and returned a successful confirmation.

If the tool fails, clearly explain that
the information was not saved.

When the student asks about their saved
information, use the get_profile tool.

Do not invent saved information.
Examples of information worth remembering:

- their name
- college or university
- major
- career goal
- long-term career interests
- programming skills
- certifications
- transfer goals
- graduation goals
- important professional experience

Do NOT save temporary or unimportant facts.

For example, do not save things like:
- what they ate today
- what time it is
- temporary homework questions
- random short-term comments

If the user changes something previously saved, update the profile.

Example:

User:
"I want to become an AI engineer."

Save:
career_goal = AI Engineer

Later:

User:
"I actually want to become a machine learning engineer."

Update:
career_goal = Machine Learning Engineer

When the user asks what you know about them, their education,
career goals, skills, interests, or background, use the get_profile
tool before answering.

You also have normal conversation memory through LangGraph.

Be helpful, practical, friendly, and clear.
"""


# =========================================================
# AGENT NODE
# =========================================================

def agent_node(state: AgentState):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# =========================================================
# GRAPH
# =========================================================

tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")


def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")


# =========================================================
# CONVERSATION MEMORY
# =========================================================

conversation_connection = sqlite3.connect(
    "agent_memory.db",
    check_same_thread=False
)

memory = SqliteSaver(conversation_connection)

app = workflow.compile(
    checkpointer=memory
)


# =========================================================
# USER / THREAD CONFIGURATION
# =========================================================

config = {
    "configurable": {
        "thread_id": CURRENT_USER
    }
}


# =========================================================
