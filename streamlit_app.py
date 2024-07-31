import streamlit as st
import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser

# _ = load_dotenv(find_dotenv())    # read local .env file
# ZHIPU_api_key=os.environ['ZHIPUAI_API_KEY']

def generate_response(input_text,ZHIPU_api_key:str):
    llm = ChatZhipuAI(temperature=0.7,api_key=ZHIPU_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    # st.info(output)
    return output

import requests

def verify_zhipu_api_key(api_key):
    url = "https://api.zhipuai.com/verify"  # 假设这是智谱 API 的验证端点
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"请求失败: {e}")
        return False

# 示例使用
api_key = "your_api_key_here"



# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
#     # if not openai_api_key.startswith('sk-'):
#     #     st.warning('Please enter your OpenAI API key!', icon='⚠')
#     # if submitted and openai_api_key.startswith('sk-'):
#     generate_response(text)


def get_vectordb(ZHIPU_api_key:str):
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings(api_key=ZHIPU_api_key)
    # 向量数据库持久化路径
    persist_directory = '../../data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

#带有历史记录的问答链
def get_chat_qa_chain(question:str,ZHIPU_api_key:str):
    vectordb = get_vectordb(ZHIPU_api_key)
    llm = ChatZhipuAI(model_name = "glm-4-flash", temperature = 0.4,api_key= ZHIPU_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str,ZHIPU_api_key:str):
    vectordb = get_vectordb(ZHIPU_api_key)
    llm = ChatZhipuAI(model_name = "glm-4-flash", temperature = 0.4,api_key = ZHIPU_api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]




# Streamlit 应用程序界面
def main():
    st.title('🤣🤞有多少人工就有多少智障💪')
    st.header('GLM-4-Flash')
    ZHIPU_api_key = st.sidebar.text_input('ZhiPu API Key', type='password')
    # if st.sidebar.button('验证'):
    #     if verify_zhipu_api_key(ZHIPU_api_key):
    #         st.success("这是一个有效的智谱 API 密钥。")
    #     else:
    #         st.error("这不是一个有效的智谱 API 密钥。")
    
    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["无RetrievalQA的普通模式", "不带memory的RetrievalQA模式", "带memory的RetrievalQA模式"])

   
    messages = st.container(height=400)
    if prompt := st.chat_input("Say something 但请先填入Api-key"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            answer = generate_response(prompt,ZHIPU_api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt,ZHIPU_api_key)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt,ZHIPU_api_key)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

        



if __name__ == "__main__":
    main()