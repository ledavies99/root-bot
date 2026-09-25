import streamlit as st
import os

from root_bot import initialize_directory_rules_bot

st.set_page_config(page_title="Root Rules Lawyer", page_icon="⚖️", layout="centered")

st.title("🌲 Root Rules Lawyer")
st.caption("v2.1 — Powered by Gemini 2.5 Flash & The Law of Root")

st.markdown("---")

st.success("🤖 **System Status:** Online. Database loaded with Core Rules, Expansion Factions, and Reddit Edge Cases.")
st.markdown("Query the official *Law of Root* and community edge-cases instantly.")

@st.cache_resource
def get_bot():
    rules_dir = "./root_rules_data"
    return initialize_directory_rules_bot(rules_dir)

try:
    bot_query_function = get_bot()
except Exception as e:
    st.error(f"Failed to initialize rules database: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Marquise, Eyrie, or Woodland Alliance? Ask me any complex rules interaction."}
    ]

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🦝"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            if "⚠️" in message["content"]:
                st.error(message["content"])
            else:
                st.info(message["content"])

if user_query := st.chat_input("Does the keep count as a building?"):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching the Law of Root..."):
            try:
                answer = bot_query_function(user_query)
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    error_msg = "⚠️ **Google Free Tier Limit Reached.** You've hit the 20 requests/day cap. Please wait a moment or switch to a pay-as-you-go key."
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    message_placeholder.error(f"An unexpected error occurred: {e}")