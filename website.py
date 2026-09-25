
import streamlit as st
import sqlite3

from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from agent_core import app, CURRENT_USER


# ==========================================
# WEBSITE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Pathway AI",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# WEBSITE DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #101727;
    color: #f0f3ff;
}

[data-testid="stSidebar"] {
    background-color: #1b2841;
}

[data-testid="stChatMessage"] {
    border-radius: 15px;
}

.stButton > button {
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background-color: #1c2b46;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #364b70;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# STUDENT PROFILE DATABASE
# ==========================================

PROFILE_DB = (
    Path(__file__).resolve().parent / "profile_memory.db"
)

PROFILE_FIELDS = [
    "name",
    "major",
    "school",
    "career_goal",
    "skills",
    "interests",
    "transfer_goal",
    "graduation_goal",
    "experience",
    "certifications"
]


def load_student_profile():

    if not PROFILE_DB.exists():
        return {}

    # Create a separate database connection for
    # reading the profile.
    with sqlite3.connect(PROFILE_DB) as connection:

        cursor = connection.execute(
            """
            SELECT field, value
            FROM student_profile
            WHERE user_id = ?
            """,
            (CURRENT_USER,)
        )

        rows = cursor.fetchall()

    return {
        field: value
        for field, value in rows
    }


# ==========================================
# CONVERSATION SESSION
# ==========================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎓 Pathway AI")

    st.caption("AI Career Pathway Counselor")

    st.divider()

    page = st.radio(
        "Workspace",
        [
            "💬 AI Counselor",
            "👤 Student Dashboard"
        ]
    )

    st.divider()

    if st.button("➕ New Chat"):

        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.write("🧠 Llama 3.2")
    st.write("💾 SQLite Memory")
    st.write("🌐 DuckDuckGo Search")

    st.divider()

    st.caption("Powered by Ollama + LangGraph")


# ==========================================
# PAGE 1: AI COUNSELOR
# ==========================================

if page == "💬 AI Counselor":

    st.title("🎓 Your future starts here.")

    st.write(
        "Explore computer science careers, "
        "discover new skills, and build your future."
    )

    st.divider()

    # Display conversation history.
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])


    # Chat input.
    prompt = st.chat_input(
        "Ask your AI career counselor..."
    )

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    config = {
                        "configurable": {
                            "thread_id":
                                st.session_state.thread_id
                        }
                    }

                    result = app.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content=prompt
                                )
                            ]
                        },
                        config=config
                    )

                    answer = result["messages"][-1].content

                    st.write(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        f"Agent error: {e}"
                    )


# ==========================================
# PAGE 2: STUDENT DASHBOARD
# ==========================================

elif page == "👤 Student Dashboard":

    st.title("👤 Student Dashboard")

    st.write(
        "Your personalized academic and career profile."
    )

    st.divider()

    # Retrieve the latest information from SQLite.
    try:

        profile = load_student_profile()

    except sqlite3.Error as e:

        st.error(f"Database error: {e}")
        st.stop()


    # Helper function for displaying fields.
    def get_value(field):

        return profile.get(
            field,
            "Not saved yet"
        )


    # Profile information.
    st.subheader("🎓 Academic & Career Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🎯 Career Goal",
            get_value("career_goal")
        )

        st.metric(
            "🏫 School",
            get_value("school")
        )

    with col2:

        st.metric(
            "📚 Major",
            get_value("major")
        )

        st.metric(
            "🎓 Transfer Goal",
            get_value("transfer_goal")
        )


    st.divider()


    # Skills and interests.
    st.subheader("💻 Skills & Interests")

    col3, col4 = st.columns(2)

    with col3:

        with st.container(border=True):

            st.subheader("Programming Skills")

            st.write(get_value("skills"))

    with col4:

        with st.container(border=True):

            st.subheader("Career Interests")

            st.write(get_value("interests"))


    st.divider()


    # Additional information.
    st.subheader("📋 Additional Information")

    with st.expander("View complete profile"):

        for field in PROFILE_FIELDS:

            label = field.replace(
                "_",
                " "
            ).title()

            st.write(
                f"**{label}:** {get_value(field)}"
            )


    # Profile completion.
    st.divider()

    st.subheader("📊 Profile Completion")

    completed = sum(
        1
        for field in PROFILE_FIELDS
        if profile.get(field)
    )

    total = len(PROFILE_FIELDS)

    completion = completed / total

    st.progress(completion)

    st.write(
        f"{completed} of {total} profile fields saved"
    )

    if not profile:

        st.info(
            "Your profile is empty. Chat with your "
            "AI counselor and tell it about your "
            "education, skills, and career goals."
        )


    # Refresh button.
    if st.button("🔄 Refresh Profile"):

        st.rerun()
