import os

import streamlit as st
from openai import OpenAIError

from knowledge_gpt.utils import get_answer_with_full_source, wrap_text_in_html, embed_docs, text_to_docs, parse_txt, \
    parse_docx, parse_pdf, search_docs, get_answer, get_sources

st.set_page_config(page_title="Docs based GPT", page_icon="📖", layout="wide")

os.environ['OPENAI_API_KEY'] = 'sk-'
st.session_state["api_key_configured"] = True
st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def intro():
    st.sidebar.success("Select a demo above.")

    st.header("📖 Docs based GPT")

    st.markdown(
        """
        ### How to use\n
        1. Upload a pdf, docx, or txt file 📄\n
        2. Ask a question about the document 💬\n
        """
    )

    st.markdown(
        """
        ---
        ### Is my data safe? \n
        Yes, your data is safe. It does not store your documents or
        questions. All uploaded data is deleted after you close the browser tab.
        """
    )


def clear_submit():
    st.session_state["submit"] = False


def text_search_with_embedding():
    st.header("📖 Chat using text search with GPT Embedding model")

    st.markdown(
        """
        ### How does this demo work?
        When you upload a document, it will be divided into smaller chunks 
        and stored in a special type of database called a vector index 
        that allows for semantic search and retrieval. When you ask a question, it will search through the
        document chunks and find the most relevant ones using the vector index.
        Then, it will use GPT-3.5 to generate a final answer.
        
        ### What do the numbers mean under each source?
        For a PDF document, you will see a citation number like this: 3-12. 
        The first number is the page number and the second number is 
        the chunk number on that page. For DOCS and TXT documents, 
        the first number is set to 1 and the second number is the chunk number.
        """
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload a pdf, docx, or txt file",
        type=["pdf", "docx", "txt"],
        help="Scanned documents are not supported yet!",
        on_change=clear_submit,
    )

    index = None
    doc = None
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = parse_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            doc = parse_docx(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            doc = parse_txt(uploaded_file)
        else:
            raise ValueError("File type not supported!")

        text = text_to_docs(doc)

        try:
            with st.spinner("Indexing document... This may take a while⏳"):
                index = embed_docs(text)
        except OpenAIError as e:
            st.error(e.message)

    query = st.text_area("Ask a question about the document", on_change=clear_submit)
    with st.expander("Advanced Options"):
        show_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
        show_full_doc = st.checkbox("Show parsed contents of the document")

    if show_full_doc and doc:
        with st.expander("Document"):
            # Hack to get around st.markdown rendering LaTeX
            st.markdown(f"<p>{wrap_text_in_html(doc)}</p>", unsafe_allow_html=True)

    button = st.button("Submit")
    if button or st.session_state.get("submit"):
        if not st.session_state.get("api_key_configured"):
            st.error("Please configure your OpenAI API key!")
        elif not index:
            st.error("Please upload a document!")
        elif not query:
            st.error("Please enter a question!")
        else:
            st.session_state["submit"] = True
            # Output Columns
            answer_col, sources_col = st.columns(2)
            sources = search_docs(index, query)

            try:
                answer = get_answer(sources, query)

                if not show_all_chunks:
                    # Get the sources for the answer
                    sources = get_sources(answer, sources)

                with answer_col:
                    st.markdown("#### Answer")
                    st.markdown(answer["output_text"].split("SOURCES: ")[0])

                with sources_col:
                    st.markdown("#### Sources")
                    for source in sources:
                        st.markdown(source.page_content)
                        st.markdown(source.metadata["source"])
                        st.markdown("---")

            except OpenAIError as e:
                st.error(e.message)


def chat_with_full_content():
    st.header('Chat using full content with ChatComplete model')

    st.markdown(
        """
        ### How does this demo work?
        When you upload a document, it will be converted to an array of string as data. When you ask a question, it will 
        use GPT-3.5 to generate a final answer by send this data as reference source.
        """
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload a pdf, docx, or txt file",
        type=["pdf", "docx", "txt"],
        help="Scanned documents are not supported yet!",
        on_change=clear_submit,
    )

    doc = None
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = parse_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            doc = parse_docx(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            doc = parse_txt(uploaded_file)
        else:
            raise ValueError("File type not supported!")

    query = st.text_area("Ask a question about the document", on_change=clear_submit)

    button = st.button("Submit")
    if button or st.session_state.get("submit"):
        if not st.session_state.get("api_key_configured"):
            st.error("Please configure your OpenAI API key!")
        elif not query:
            st.error("Please enter a question!")
        else:
            st.session_state["submit"] = True

            try:
                answer = get_answer_with_full_source(docs=' '.join(doc), question=query)

                st.markdown("#### Answer")
                st.markdown(answer)

            except OpenAIError as e:
                st.error(e.message)


page_names_to_funcs = {
    "—": intro,
    "Text search in vector db": text_search_with_embedding,
    "Chat with full content": chat_with_full_content,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
