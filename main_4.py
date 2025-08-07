from http.client import responses

import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils_4 import qa_agent


st.title("AI智能PDF问答工具")
with st.sidebar:
    qwen_api_key = st.text_input("请输入你的通义千问API密钥", type="password")
    st.markdown("[获取通义千问API KEY](https://www.aliyun.com/product/pai)")

if "memory" not in st.session_state:# 初始化对话记忆
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
# 文件上传区
uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")
question = st.text_input("对PDF的内容进行提问", disabled=not uploaded_file)# 问题输入区

if uploaded_file and question and not qwen_api_key:
    st.info("请输入你的通义千问API密钥")
# 处理问答流程
if uploaded_file and question and qwen_api_key:
    with st.spinner("AI正在思考中，请稍等..."):
        response = qa_agent(qwen_api_key, st.session_state["memory"], uploaded_file, question)
    st.write("### 答案")
    st.write(response["answer"])
    # 保存对话历史
    st.session_state["chat_history"] = response["chat_history"]
# 显示历史对话
if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
