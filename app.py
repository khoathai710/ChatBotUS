import streamlit as st
from model_structure import ChatBotUS  
import pickle
import fitz

st.set_page_config(page_title='ChatBot US', page_icon='💬')
st.markdown("<h1 style='text-align: center;'>ChatBot cho Ú-er</h1>", unsafe_allow_html=True)

with st.chat_message("assistant", avatar="data/background.png"):
    st.markdown("Xin chào mình là trợ lí AI phục vụ cho sinh viên trường Khoa học Tự nhiên. Mình có thể giúp gì cho bạn?")

if "messages" not in st.session_state:
    st.session_state.messages = []

    
if 'model' not in st.session_state:
    st.session_state.model = ChatBotUS()
    
if 'type' not in st.session_state:
    st.session_state.type = '📕Sổ Tay Sinh Viên'

for message in st.session_state.messages:
    
    if message["role"] == "user":
        st.markdown(f"""
    <div style="text-align: left; background-color: #262730; padding: 10px; 
                border-radius: 10px; width: fit-content; max-width: 70%;
                margin-left: auto; display: block; color: white;">
        {message["content"]}
    </div>
    """, unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="data/background.png"):
            st.markdown(message["content"])
            
model = ChatBotUS()

with st.sidebar:
    database = st.radio("Hãy chọn nguồn", ["📕Sổ Tay Sinh Viên", "📄PDF",'🔗URL'])
    
    
        
    if database == '📕Sổ Tay Sinh Viên':
        if "model" not in st.session_state or st.session_state.type != database:
            st.session_state.model = ChatBotUS()
            st.session_state.type = database
            
    if database == "📄PDF": 
        uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")

        if uploaded_file is not None:
            #

            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            
            text = ""
            
            for page in doc:
                text += page.get_text("text") + "\n"
            
            if text != '':
                if "model" not in st.session_state or st.session_state.type != database:
                    st.session_state.model = ChatBotUS(text=text)
                    st.session_state.type = database

                
            
        
    with st.expander("📜 Lịch sử", expanded=True):
        st.markdown('')
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown('- ' + message['content'])
                    
if prompt := st.chat_input("Nhập câu hỏi của bạn"):
    
    st.markdown(f"""
    <div style="text-align: left; background-color: #262730; padding: 10px; 
                border-radius: 10px; width: fit-content; max-width: 70%;
                margin-left: auto; display: block;color: white;">
        {prompt}
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": prompt})

    model = st.session_state.model
    with st.chat_message("assistant", avatar="data/background.png"):
        with st.spinner("Chờ mình tí..."):
            try:
                response = model.make_response(question=prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi tạo phản hồi: {e}")
