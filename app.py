import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import os

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(page_title="钛合金专家系统 AI", page_icon="⚙️", layout="wide")

st.title("⚙️ 钛合金/特种金属 AI 专家系统")
st.markdown("### 基于 DeepSeek-V3 与 工业国标数据库")

# ==========================================
# 2. 侧边栏：配置与数据加载
# ==========================================
with st.sidebar:
    st.header("🔧 系统配置")
    
    # 让这个程序变得通用：用户可以自己填 Key，也可以用默认的
    user_api_key = st.text_input("输入 DeepSeek API Key", value="", type="password")
    
    st.divider()
    
    st.subheader("📂 数据库状态")
    # 自动加载当前目录下的 CSV
    try:
        if os.path.exists("titanium_composition.csv") and os.path.exists("titanium_properties.csv"):
            df_comp = pd.read_csv("titanium_composition.csv")
            df_prop = pd.read_csv("titanium_properties.csv")
            st.success(f"✅ 成分表已加载: {len(df_comp)} 条")
            st.success(f"✅ 性能表已加载: {len(df_prop)} 条")
            
            # 可以在侧边栏预览数据
            with st.expander("查看原始数据预览"):
                st.write("成分表:", df_comp.head(3))
                st.write("性能表:", df_prop.head(3))
        else:
            st.error("❌ 未找到 CSV 文件，请确保文件在同一目录下！")
            st.stop()
    except Exception as e:
        st.error(f"数据读取失败: {e}")
        st.stop()

    st.divider()
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. 初始化 Agent (带缓存，防止每次提问都重载)
# ==========================================
@st.cache_resource
def get_agent(api_key):
    if not api_key:
        return None
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.1,
        max_tokens=2048
    )

    PREFIX_PROMPT = """
    你是一个精准的钛合金数据专家。
    【数据字典】
    1. df1 (成分表): Grade(牌号), Al, Fe, ...
    2. df2 (性能表): Grade(牌号), State(状态), Rm(强度), A(延伸率)
    
    【回答规则】
    1. **查数据**：必须用 Python 查表。如果查到数据，请尽量用 Markdown 表格形式输出。
    2. **通用知识**：直接回答。
    3. **无记录**：请明确告知。
    """

    return create_pandas_dataframe_agent(
        llm,
        [df_comp, df_prop],
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        prefix=PREFIX_PROMPT,
        handle_parsing_errors=True
    )

agent = get_agent(user_api_key)

# ==========================================
# 4. 聊天主界面逻辑
# ==========================================

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的专属材料专家。您可以问我关于 TC4、TA1 等钛合金的成分、强度或热处理工艺。"}]

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("请输入您的问题 (例如: 帮我找强度大于1000的牌号)"):
    # 1. 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 正在思考并查询数据库...")
        
        try:
            if not agent:
                response = "❌ 请先在侧边栏输入有效的 API Key。"
            else:
                # 调用 LangChain Agent
                result = agent.invoke({"input": prompt})
                response = result["output"]
            
            message_placeholder.markdown(response)
            
            # 3. 保存回答到历史
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"⚠️ 发生错误: {str(e)}"
            message_placeholder.error(error_msg)
