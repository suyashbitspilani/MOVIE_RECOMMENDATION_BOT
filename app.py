import streamlit as st
# st.set_page_config(
#     page_title="Hello",
#     page_icon="👋",
# )
# st.write("# Welcome to Streamlit! 👋")
#
#
# st.markdown("""
# This component supports **markdown formatting**, which is handy.
#
# [Check out their documentation](https://docs.streamlit.io) for more information on how to get started.
# """)



# from IPython.display import Markdown
#
#
# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

GOOGLE_API_KEY="AIzaSyBt6CkN4Toq16DOtpUclW2lDXkdcvB-Y0A"

from langchain_google_genai import GoogleGenerativeAI
# llmm= GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY)
# print(
#     llmm.invoke(
#         "What are some of the pros and cons of Python as a programming language?"
#     )
# )
# llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
#
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
#
# print(
#     llm.invoke(
#         "What are some of the pros and cons of Python as a programming language?"
#     )
# )
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
)
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)
