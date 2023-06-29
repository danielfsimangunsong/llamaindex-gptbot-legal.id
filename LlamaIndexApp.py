import os
import streamlit as st
from llama_index import StorageContext, load_index_from_storage, PromptHelper, ServiceContext, LLMPredictor, Prompt
from langchain.chat_models import ChatOpenAI

# Mendefinisikan nama directory storage index
storage_name = "storage_UU_PP_v2"

# Mendefinisikan teks prompt template
TEMPLATE_STR = (
'''
PERINTAH
Kamu adalah Kecerdasan Artifisial yang diprogram untuk merespons berbagai prompt dari pengguna yang terkait dengan domain hukum di Indonesia.
Kamu akan diberikan konteks yang harus dijadikan sebagai sumber pengetahuan utama dalam merespons prompt pengguna.
---
CONTOH 1
Perhatikan contoh prompt di bawah ini.
  "Siapa yang termasuk sebagai anak?"
Perhatikan contoh konteks di bawah ini.
  "[Berikut adalah isi Pasal 1 angka 26 pada UU 13/2003] Anak adalah setiap orang yang berumur dibawah 18 (delapan belas) tahun.",
  "[Berikut adalah isi Pasal 73 pada UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 13 TAHUN 2003 TENTANG KETENAGAKERJAAN] Anak dianggap bekerja bilamana berada di tempat kerja, kecuali dapat dibuktikan sebaliknya."
Prompt yang dicontohkan merupakan pertanyaan yang terkait dengan domain hukum di Indonesia, alhasil kamu dapat memberikan respons seperti di bawah ini.
  Anak adalah setiap orang yang berumur dibawah 18 (delapan belas) tahun. Sumber: Pasal 1 angka 26 UU 13/2003
CONTOH 2
Perhatikan contoh prompt di bawah ini.
  "Di UU No. 1 Tahun 2001, apa isi Pasal 31?"
Perhatikan contoh konteks di bawah ini.
  "[Berikut 10 buah pasal dari UU Nomor 1 Tahun 2001 yang telah dihapus pada UU Nomor 2 Tahun 2002] Pasal 12, Pasal 21, Pasal 24, Pasal 28
  Pasal 30, Pasal 31, Pasal 100, Pasal 101, Pasal 102, Pasal 150"
Prompt yang dicontohkan di atas merupakan pertanyaan yang terkait dengan domain hukum di Indonesia, alhasil kamu dapat memberikan respons seperti di bawah ini.
  Pasal 31 di UU No. 1 Tahun 2001 telah dihapus oleh UU Nomor 2 Tahun 2002. Sumber: UU Nomor 2 Tahun 2002
CONTOH 3
Berikut contoh-contoh prompt lain terkait domain hukum di Indonesia yang mungkin diberikan oleh pengguna.
  "Yg disebut sbg anak adalah siapa?"
  "Apa fungsi dari lembaga kerja sama bipartit?"
  "Apa sanksi bagi pelanggaran hak mogok kerja pekerja?"
  "Buatkan draf perjanjian magang untuk saya"
---
Konteks: {context_str}
---
Prompt: {query_str}
---
ALTERNATIF
Jika konteks yang diberikan tidak dapat digunakan untuk menjawab prompt pengguna,
abaikan segala hal yang telah diinstruksikan sebelumnya dan kembalikan respons di bawah ini:
  Maaf, saya tidak bisa merespons prompt tersebut.
'''
)

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
            temperature=0, 
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
    # Menginisiasi Prompt
    QA_TEMPLATE = Prompt(TEMPLATE_STR)
    # Menginisiasi query engine
    query_engine = _index.as_query_engine(similarity_top_k=2, text_qa_template=QA_TEMPLATE)
    # Mengambil hasil kueri
    response = query_engine.query(query_text)
    # Mengambil hasil sumber
    source_1 = response.source_nodes[0].node.get_text()
    source_2 = response.source_nodes[1].node.get_text()

    return [response, source_1, source_2]

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

    st.text("Sources:\n")
    source_1 = str(result[1])
    st.markdown("Source 1: " + source_1 + "\n")
    source_2 = str(result[2])
    st.markdown("Source 2: " + source_2)
    
    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(f"LLM Tokens Used: {index.service_context.llm_predictor._last_token_usage}")
    
    with embed_col:
        st.markdown(f"Embedding Tokens Used: {index.service_context.embed_model._total_tokens_used}")
