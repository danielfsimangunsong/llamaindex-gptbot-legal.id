import os
import streamlit as st
from llama_index import StorageContext, load_index_from_storage, PromptHelper, ServiceContext, LLMPredictor
from langchain.chat_models import ChatOpenAI

# Mendefinisikan nama directory storage index
storage_name = "storage_UU_PP"

@st.cache_resource
def initialize_index(storage_name):
    if os.path.exists(storage_name):
        # Membangun ulang storage context
        storage_context = StorageContext.from_defaults(persist_dir=storage_name)
       
        # Mendefinisikan prompt helper
        context_window = 4096
        num_output = 256
        prompt_helper = PromptHelper(context_window=context_window, num_output=num_output)

        # Mendefinisikan LLM
        llm_predictor = LLMPredictor(llm=ChatOpenAI(
            temperature=0.1, 
            model_name="gpt-3.5-turbo", 
            max_tokens=num_output)
        )

        # Mendefinisikan service context
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            prompt_helper=prompt_helper,
        )

         # Memuat index
        loaded_index = load_index_from_storage(storage_context, service_context=service_context)
        return loaded_index
    else:
        st.warning("Please provide a saved index first.")
        return None

@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    # Menginisiasi query engine
    query_engine = _index.as_query_engine(similarity_top_k=2)
    # Mengambil hasil kueri
    response = query_engine.query(query_text)
    # Mengambil hasil sumber
    source = response.source_nodes[0].node.get_text()

    return [response, source]

# Mengatur tampilan aplikasi
st.title("🦙 LlaLex Bot ⚖️ ")
st.write("Enter a query about Indonesian Labor Laws. Your query will be answered using the Laws Document as context, using embeddings from text-ada-002, and LLM completions from GPT-3.5.")

# Menyiapkan index
index = None
api_key = st.text_input("Enter your OpenAI API key here:", type="password")
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    index = initialize_index(storage_name)    
if index is None:
    st.warning("Please enter your api key first.")

# Melakukan kueri
text = st.text_input("Query text:", value="Apa definisi dari kompetensi kerja?")

# Mengatur tampilan hasil kueri
if st.button("Run Query") and text is not None:
    result = query_index(index, text)
    response = str(result[0])
    print(response)
    st.text("Answer:\n")
    st.markdown(response)

    source = str(result[1])
    st.text("Source:\n")
    st.markdown(source)
    
    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(f"LLM Tokens Used: {index.service_context.llm_predictor._last_token_usage}")
    
    with embed_col:
        st.markdown(f"Embedding Tokens Used: {index.service_context.embed_model._total_tokens_used}")
