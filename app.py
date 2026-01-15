import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
import os

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(page_title="钛合金专家系统 AI", page_icon="⚙️", layout="wide")

st.title("⚙️ 钛合金/特种金属 AI 专家系统")
st.markdown("### 基于 DeepSeek-V3 与 天元新材数据库")

# ==========================================
# 2. 数据加载 (使用 session_state 持久化)
# ==========================================
@st.cache_data
def load_data():
    """加载 CSV 数据，使用 cache_data 确保数据一致性"""
    if os.path.exists("titanium_composition_cleaned.csv") and os.path.exists("titanium_properties_cleaned.csv"):
        df1 = pd.read_csv("titanium_composition_cleaned.csv")
        df2 = pd.read_csv("titanium_properties_cleaned.csv")
        return df1, df2
    return None, None

# 加载数据到 session_state（只在首次运行时加载）
if "df_comp" not in st.session_state or "df_prop" not in st.session_state:
    df_comp, df_prop = load_data()
    if df_comp is not None:
        st.session_state.df_comp = df_comp
        st.session_state.df_prop = df_prop

# 从 session_state 获取数据
df_comp = st.session_state.get("df_comp")
df_prop = st.session_state.get("df_prop")

# ==========================================
# 3. 侧边栏：配置与状态显示
# ==========================================
with st.sidebar:
    st.header("🔧 系统配置")

    # 让这个程序变得通用：用户可以自己填 Key，也可以用默认的
    user_api_key = st.text_input("输入 DeepSeek API Key", value="sk-664268bc084c4a3fbd19fbb9efc924da", type="password")

    st.divider()

    st.subheader("📂 数据库状态")
    if df_comp is not None and df_prop is not None:
        st.success(f"✅ 成分表已加载: {len(df_comp)} 条")
        st.success(f"✅ 性能表已加载: {len(df_prop)} 条")

        # 可以在侧边栏预览数据
        with st.expander("查看原始数据预览"):
            st.write("成分表:", df_comp.head(3))
            st.write("性能表:", df_prop.head(3))
    else:
        st.error("❌ 未找到 CSV 文件，请确保文件在同一目录下！")
        st.stop()

    st.divider()
    st.write("🕵️‍♂️ 数据透视自检:")
    # 模糊搜索 TC4
    check = df_comp[df_comp['Grade'].str.contains("TC4", case=False, na=False)]
    if not check.empty:
        st.success(f"✅ 内存中存在 TC4 数据！(共{len(check)}条)")
        st.dataframe(check)
    else:
        st.error("❌ 严重警告：内存中的 DataFrame 里真的没有 TC4！")

    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        # 同时清空 Agent 缓存，确保下次使用最新数据
        if "agent" in st.session_state:
            del st.session_state.agent
        st.rerun()

# ==========================================
# 4. 初始化 Agent (使用 session_state 缓存)
# ==========================================
def create_agent(api_key, df1, df2):
    """创建 Agent，不使用 @st.cache_resource 避免 DataFrame 缓存问题"""
    if not api_key:
        return None

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0,
        max_tokens=4096
    )

    # 1. 动态获取列名，直接告诉 AI
    df1_columns = ", ".join(df1.columns.tolist())
    df2_columns = ", ".join(df2.columns.tolist())

    # 2. 构建包含“上帝视角”的 Prompt
    PREFIX_PROMPT = f"""
    你是一个精准的钛合金数据专家。数据已经加载到内存变量 df1 和 df2 中。
    
    【严禁操作】
    1. 严禁使用 pd.read_csv 读取文件！直接使用 df1 和 df2。
    2. 严禁运行代码去查看 df.head() 或 df.columns，因为下面已经告诉你了。
    
    【数据结构说明（已直接提供，无需查询）】
    1. df1 (成分表) 包含列: [{df1_columns}]
       - 用途：查询化学成分（如 Al, V, Fe, C, N 等元素含量）。
       - 关键列：'Grade' 是牌号。
       
    2. df2 (性能表) 包含列: [{df2_columns}]
       - 用途：查询物理/机械性能（如 Rm 抗拉强度, Rp0.2 屈服强度, A 延伸率, Z 断面收缩率等）。
       - 关键列：'Grade' 是牌号。
    
    【回答原则】
    1. 用户查牌号时，优先通过 'Grade' 列关联 df1 和 df2。
    2. 必须严格基于 Python 工具运行返回的数据回答。
    3. 如果数据中包含范围（如 5.5~6.5），请完整展示。

    【⭐⭐ 核心搜索法则 (必须严格遵守，覆盖默认行为) ⭐⭐】
    为了防止漏查，当用户查询某个牌号（如 TC4, SP700）时，你必须遵守以下逻辑：

    1. 🚫 **绝对禁止** 使用 `df['Grade'].unique()`、`.values` 或 `.head()` 来肉眼寻找牌号。
    2. 🚫 **绝对禁止** 在代码中对搜索结果进行切片（例如 `[:20]` 或 `head(5)`）。即使数据量大，也必须让 Pandas 完整返回，不要担心 Token 消耗。
    3. ✅ **必须** 直接运行全量模糊搜索代码。
       - 标准代码范例：`df1[df1['Grade'].str.contains('SP700', case=False, na=False)]`
    4. 🔄 **自动重试机制**：
       - 如果第一次搜索 `df1` 返回 Empty，不要立刻说找不到。
       - 必须紧接着搜索 `df2`。
       - 必须尝试变体搜索（例如用户搜 'Ti64' 没搜到，尝试搜 'Ti-64' 或 'Ti 64'）。
    
    只有当所有模糊匹配的代码运行结果都为空时，才能回复“未找到”。
    """
    return create_pandas_dataframe_agent(
        llm,
        [df1, df2],
        verbose=False,
        allow_dangerous_code=True,
        agent_type="openai-tools",  # <--- 核心修改：改为 tools 模式
        prefix=PREFIX_PROMPT,
        agent_executor_kwargs={
            "handle_parsing_errors": True  # 自动处理简单的解析错误
        }
    )

# 使用 session_state 缓存 Agent（仅在 API Key 变化或首次运行时创建）
if df_comp is not None and df_prop is not None:
    # 检查是否需要重新创建 Agent
    need_new_agent = (
        "agent" not in st.session_state or
        st.session_state.get("agent_api_key") != user_api_key
    )

    if need_new_agent:
        # 传入副本，防止 agent 执行代码时修改原始数据
        st.session_state.agent = create_agent(user_api_key, df_comp.copy(), df_prop.copy())
        st.session_state.agent_api_key = user_api_key

    agent = st.session_state.agent
else:
    st.warning("数据未加载，请检查 CSV 文件")
    agent = None

# ==========================================
# 4. 聊天主界面逻辑
# ==========================================

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的专属材料专家。您可以让我查询 TC4、TA1、SP700 等钛合金的成分及性能数据。"}]

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("请输入您的问题 (例如: 帮我找适合制造高尔夫球头的钛合金牌号)"):
    # 1. 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回答
    with st.chat_message("assistant"):
        # 创建一个空的容器，用于放置“思考过程”
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            if not agent:
                response = "❌ 请先在侧边栏输入有效的 API Key。"
            else:
                # 关键修改：传入 callbacks
                # 这样 Agent 的每一步动作都会实时打印在屏幕上
                result = agent.invoke(
                    {"input": prompt},
                    config={"callbacks": [st_callback]}
                )
                
                response = result["output"]
                st.markdown(response)
                
                # 3. 保存回答到历史
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error(f"⚠️ 发生错误: {str(e)}")