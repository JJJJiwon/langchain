import streamlit as st  # 파이썬 스크립트를 구동하기 위함
import tiktoken    # text를 여러개의 청크로 나눌때 문자의 갯수를 무엇을 기준으로 산정을 할거냐 : 토큰갯수로 셀거야!
from loguru import logger  # streamlit에서 구동한 것들이 로그로 남도록 함

from langchain.chains import ConversationalRetrievalChain  # 메모리를 가지고 있는 체인...?
from langchain.chat_models import ChatOpenAI   

# 여러 유형의 문서를 넣어도 모두 이해 시킬수 있게 하는 것들: 워드 피피티...등등
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter  # text를 나눌때 사용
from langchain.embeddings import HuggingFaceEmbeddings  # 한국어에 특화된 허깅페이스에 임베딩 모델 사용
  
from langchain.memory import ConversationBufferMemory  # 몇개까지의 대화를 내가  메모리에 넣어줄지 결정
from langchain.vectorstores import FAISS  # 임시로 벡터를 저장하기 위함

# 메모리를 구현하기 위한 추가적인 라이브러리
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


def main():
    st.set_page_config(
    page_title="jiwon",
    page_icon=":books:")  # 아이콘은 변경 가능

    st.title("Jiwon_ChatBot :red[QA Chat]_ :books:")  # _ : 기울임
    
    # 이후 파이썬 스크립트에서 "session_state.conversation"이라는 변수를 쓰기위해 먼저 정의해줌
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    
    # 사이드바 부분 코드
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    
    # 만약에 process 버튼을 누르면 
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)  # 업로드된 파일들을 TEXT로 변환
        text_chunks = get_text_chunks(files_text) # 어려개의 TEXT 청크로 나눔
        vetorestore = get_vectorstore(text_chunks) # 벡터화
        
        # 이 벡터를 LLM이 답변을 할 수 있도록 chain을 구성
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 
        
        st.session_state.processComplete = True
    
    # 채팅화면을 구현하기 위한 코드 
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]   # 채팅창에 초기값 설정

    # 메세지들을 with구문으로 묶어줌 -> 메세지의 role에 따라서 메세지의 content를 마크다운 할거다...?
    # 메세지가 입력이 될때마다 화면상에 구현
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    # 질문창
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)
       
        # 답변해주는 부분 
        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']   # 참고한 문서

                st.markdown(response)
                with st.expander("참고 문서 확인"):  # expander : 접었다가 폈다가 하는 부분
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# 업로드된 파일들을모두 text로 변환하는 함수
def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

# 여러개의 청크들로 split
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

# 청크를 벡터화
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}  # 벡터저장소에 저장해서 사용자의 질문과 비교하기 위함
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

# 위에서 선언한 모든것들을 담는 함수
def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
