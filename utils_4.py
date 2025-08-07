#基于用户上传的文档回答用户的问题 智能pdf问答工具
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader#用于加载pdf文件内容
from langchain_community.embeddings import DashScopeEmbeddings#用于将文本转换为向量表示（嵌入）
from langchain_community.vectorstores import FAISS#向量数据库
from langchain_community.chat_models import ChatTongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter#用于将长文本分割成较小的块


def qa_agent(qwen_api_key, memory, uploaded_files, question):
    model = ChatTongyi(model = "qwen-turbo", api_key=qwen_api_key)
    file_content = uploaded_files.read()#读取上传的PDF文件内容，并保存到临时文件中。
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)#加载PDF文件的内容
    docs = loader.load()
    #开始拆分文档，分割成多个小块，每个块大约1000个字符，相邻块之间有50个字符重叠。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。" "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    #开始嵌入
    embeddings_model = DashScopeEmbeddings()# 创建一个嵌入模型实例。文本向量化（将文字转换为数学向量）
    #开始向量化
    db = FAISS.from_documents(texts, embeddings_model)# 创建向量数据库（存储文本片段）
    #导入检索器
    retriever = db.as_retriever()#创建一个检索器，用于从向量数据库中查找最相关的文档片段。
    #创建检索器的链，创建一个对话式检索链，结合聊天模型和检索器和记忆。
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question":question})
    return response
