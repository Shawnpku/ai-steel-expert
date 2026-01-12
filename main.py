import sys
import io
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# ==========================================
# 0. 【核心修复】强制设置编码 (解决 UnicodeDecodeError)
# ==========================================
# 这两行代码会强制 Python 用 UTF-8 格式处理输入输出，防止中文报错
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 1. 配置区域
# ==========================================
DEEPSEEK_API_KEY = "sk-*****" 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ==========================================
# 2. 加载数据
# ==========================================
print("🔄 正在启动钛合金专家系统...")
try:
    df_comp = pd.read_csv("titanium_composition.csv")
    df_prop = pd.read_csv("titanium_properties.csv")
    print(f"✅ 数据加载成功 | 成分表:{len(df_comp)}条 | 性能表:{len(df_prop)}条")
except Exception as e:
    print(f"❌ 错误: 无法读取CSV文件。\n原因: {e}")
    exit()

# ==========================================
# 3. 提示词
# ==========================================
PREFIX_PROMPT = """
你是一个精准的钛合金数据专家。

【数据字典】
1. df1 (成分表): Grade(牌号), Al, Fe, ...
2. df2 (性能表): Grade(牌号), State(状态), Rm(强度), A(延伸率)

【回答规则】
1. **优先查表**：遇到查数据问题（如"TC4强度"），必须查表。
2. **通用知识**：遇到通用问题（如"TC4的物理特性"、"什么是退火"），直接用你的知识库回答，**不要**查表。
3. **查不到**：如果查表后 DataFrame 为空，请说“数据库无记录”。
"""

# ==========================================
# 4. 初始化 Agent
# ==========================================
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.1,
    max_tokens=2048
)

agent = create_pandas_dataframe_agent(
    llm,
    [df_comp, df_prop],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    prefix=PREFIX_PROMPT,
    handle_parsing_errors=True,
    max_iterations=5
)

# ==========================================
# 5. 交互循环
# ==========================================
def chat_loop():
    print("\n💡 系统已就绪！可以输入中文提问了 (输入 exit 退出)")
    print("---------------------------------------")
    
    while True:
        try:
            # 使用最简单的提示符，避免额外干扰
            print("\n>>> ", end="", flush=True)
            user_input = sys.stdin.readline().strip() # 使用 sys.stdin 读取，比 input() 更稳健
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("👋 再见！")
                break
            
            # 调用 Agent
            response = agent.invoke({"input": user_input})
            print(f"🤖 回答: {response['output']}")
            
        except Exception as e:
            print(f"\n⚠️ 发生错误: {e}")
            print("请重试...")

if __name__ == "__main__":
    chat_loop()
