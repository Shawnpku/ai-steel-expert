import os
from dotenv import load_dotenv

# 1. 加载 .env 里的 API Key
load_dotenv()

# 检查 key 是否加载成功
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 错误：未找到 API Key，请检查 .env 文件")
    exit()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# === A. 准备数据 (以后这里换成读取 PDF 的代码) ===
print("🔄 正在初始化数据...")
steel_data = [
    {
        "name": "316L不锈钢",
        "content": "牌号：316L。化学成分：铬16-18%，镍10-14%，钼2-3%。特性：含钼，耐海洋和化工腐蚀，抗氯离子腐蚀能力优于304。应用：食品工业、海洋设备。",
    },
    {
        "name": "2205双相钢",
        "content": "牌号：2205 (S31803)。化学成分：铬22%，镍5%，钼3%，氮0.17%。特性：双相结构，强度是316L的两倍，极好的抗应力腐蚀开裂。应用：高氯环境、海水淡化。",
    },
    {
        "name": "TA2纯钛 (Gr2)",
        "content": "牌号：TA2 (ASTM Gr2)。化学成分：工业纯钛。特性：优异的耐海水腐蚀性，密度低(4.51)，塑性好，易焊接。应用：海水换热器、电镀设备。强度适中。",
    },
    {
        "name": "TC4钛合金 (Gr5)",
        "content": "牌号：TC4 (ASTM Gr5)。化学成分：Ti-6Al-4V。特性：强度极高，但塑性差，难变形，焊接需保护。应用：航空航天、高强度结构件。不建议用于这就换热扩管。",
    }
]

# 把数据变成对象
docs = [Document(page_content=d["content"], metadata={"name": d["name"]}) for d in steel_data]

# === B. 初始化模型 (自动读取环境变量) ===
# 注意：DeepSeek 兼容 OpenAI 协议，所以这里依然用 ChatOpenAI 类
llm = ChatOpenAI(
    model="deepseek-chat", # 这里填模型名字，如果是 DeepSeek V3 就填 deepseek-chat
    temperature=0
)

embeddings = OpenAIEmbeddings(
    # DeepSeek 目前没有Embedding模型，这里我们通常用 OpenAI 的 text-embedding-3-small
    # 或者用本地的 HuggingFaceEmbeddings (为了简单，先假设你有 OpenAI key 用来做 embedding，或者 DeepSeek 将来支持)
    # *注：为了让你跑通，如果你只有 DeepSeek Key，这里可能会报错。
    # 临时解决方案：DeepSeek 用户通常搭配一个免费的 embedding 模型，或者只需少量 OpenAI 额度*
)

# === C. 向量化存入本地数据库 ===
# persist_directory 是数据库存硬盘的文件夹
vector_db = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("✅ 知识库构建完成！")

# === D. 定义专家逻辑 ===
PROMPT_TEMPLATE = """
你是一位特种金属材料专家。基于以下上下文(Context)回答问题。
如果上下文里没有答案，就说“资料库里没查到”。

【资料库数据】：
{context}

【客户问题】：
{question}
"""

def ask_expert(question):
    # 1. 检索：去数据库找最相关的2条
    results = vector_db.similarity_search(question, k=2)
    
    # 2. 拼凑上下文
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    # 3. 提问
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    
    response = chain.invoke({"context": context_text, "question": question})
    return response.content

# === E. 运行测试 ===
if __name__ == "__main__":
    print("\n💬 正在咨询 AI 专家...\n")
    
    q1 = "我要做海水换热器，用TC4还是TA2？"
    print(f"问：{q1}")
    print(f"答：{ask_expert(q1)}")
    
    print("-" * 30)
    
    q2 = "316L里含有什么成分？"
    print(f"问：{q2}")
    print(f"答：{ask_expert(q2)}")
