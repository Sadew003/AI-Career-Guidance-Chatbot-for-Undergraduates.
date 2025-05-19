import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load career knowledge base
def load_career_data():
    career_data = {
        "Computer Science": {
            "careers": ["Software Engineer", "Data Scientist", "Cybersecurity Analyst"],
            "skills": ["Python", "Java", "Data Analysis", "Cybersecurity"],
            "resources": ["Coursera", "LeetCode", "TryHackMe"]
        },
        "Business": {
            "careers": ["Marketing Manager", "Financial Analyst", "Entrepreneur"],
            "skills": ["Marketing", "Finance", "Leadership"],
            "resources": ["LinkedIn Learning", "Coursera", "Harvard Business Review"]
        },
        "Biology": {
            "careers": ["Biomedical Researcher", "Environmental Scientist", "Pharmacist"],
            "skills": ["Research", "Lab Techniques", "Data Analysis"],
            "resources": ["PubMed", "Khan Academy", "Nature Journal"]
        }
    }
    return career_data

# Create a prompt template for career guidance
def get_prompt_template():
    template = """
You are a career guidance chatbot designed for undergraduates. Your goal is to provide personalized, concise, and actionable career advice based on the user's academic major, interests, and skills. Use the following knowledge base for reference:

{career_data}

Conversation history:
{chat_history}

User input: Major: {major}\nInput: {user_input}

Analyze the user's major and input to suggest relevant career paths, skills to develop, and learning resources. If the major is not in the knowledge base, reason about related fields and provide plausible suggestions. Keep responses friendly, professional, and under 200 words.
"""
    return ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Major: {major}\nInput: {user_input}")
    ])

# Main chatbot function
def chatbot_response(user_input, major, runnable, career_data, session_id="default"):
    # Format career data for the prompt
    career_data_str = json.dumps(career_data, indent=2)
    
    # Get response from the runnable with history
    response = runnable.invoke(
        {"career_data": career_data_str, "major": major, "user_input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

# Get or create chat history for a session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

# Streamlit app
def main():
    st.title("AI Career Guidance Chatbot for Undergraduates")
    st.write("Enter your academic major and ask for career advice!")

    # Initialize session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load career data
    career_data = load_career_data()

    # Get Gemini API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Gemini API key not found in .env file. Please enter it below.")
        api_key = st.text_input("Enter your Gemini API key:", type="password")
        if not api_key:
            st.error("Gemini API key is required to proceed.")
            return

    # Initialize LangChain with Gemini 1.5 Flash
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        prompt = get_prompt_template()
        runnable = prompt | llm
        runnable_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="user_input",
            history_messages_key="chat_history"
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {str(e)}")
        return

    # User input for major
    major = st.text_input("Enter your academic major (e.g., Computer Science, Business, Biology):")

    # Chat input
    user_input = st.text_input("Ask a career-related question:")

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_input and major:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get chatbot response
        try:
            response = chatbot_response(user_input, major, runnable_with_history, career_data)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()