import pandas as pd
import os

def clean_titanium_data():
    print("🚀 开始清洗数据...")
    
    # 1. 加载数据
    if not os.path.exists("titanium_composition.csv") or not os.path.exists("titanium_properties.csv"):
        print("❌ 错误：未找到 CSV 文件，请确保文件在当前目录下。")
        return

    df_comp = pd.read_csv("titanium_composition.csv")
    df_prop = pd.read_csv("titanium_properties.csv")

    # 2. 基础清洗：去空格
    df_comp.columns = df_comp.columns.str.strip()
    df_prop.columns = df_prop.columns.str.strip()
    df_comp['Grade'] = df_comp['Grade'].astype(str).str.strip()
    df_prop['Grade'] = df_prop['Grade'].astype(str).str.strip()

    # 3. 映射表：将日标/美标映射回国标 (解决 TC4 vs Ti-6Al-4V 问题)
    grade_map = {
        'Ti-6Al-4V': 'TC4',
        # 'TP270': 'TA1',
        # 'TP340': 'TA2',
        # 'TP480': 'TA3',
        # 'TP550': 'TA4',
        'Ti-3Al-2.5V': 'TA18',
        'Ti-4Al-22V': 'TC18' # 常见对应，视具体情况而定
    }
    
    print(f"🔄 执行牌号映射: {grade_map}")
    df_comp['Grade'] = df_comp['Grade'].replace(grade_map)
    df_prop['Grade'] = df_prop['Grade'].replace(grade_map)

    # 4. 成分表合并策略
    def agg_comp(x):
        first = x.iloc[0].copy()
        # 合并备注信息
        if 'Comments' in x.columns:
            unique_comments = x['Comments'].dropna().unique()
            first['Comments'] = ' | '.join(unique_comments)
        return first

    print("🧩 合并成分表重复项...")
    df_comp_clean = df_comp.groupby('Grade', as_index=False).apply(agg_comp).reset_index(drop=True)

    # 5. 性能表合并策略
    def agg_prop(x):
        base = x.iloc[0].copy()
        # 合并文本列
        for col in ['Process', 'Usage']:
            if col in x.columns:
                unique_vals = x[col].dropna().unique()
                base[col] = ' | '.join(unique_vals)
        return base

    print("🧩 合并性能表重复项...")
    # 按牌号和状态分组 (避免把退火态和固溶态合并了)
    df_prop_clean = df_prop.groupby(['Grade', 'State'], as_index=False).apply(agg_prop).reset_index(drop=True)

    # 6. 保存文件
    # 覆盖原文件前，建议先备份，或者保存为新文件名
    df_comp_clean.to_csv("titanium_composition_cleaned.csv", index=False)
    df_prop_clean.to_csv("titanium_properties_cleaned.csv", index=False)
    
    print(f"✅ 清洗完成！")
    print(f"生成文件: titanium_composition_cleaned.csv ({len(df_comp_clean)} 行)")
    print(f"生成文件: titanium_properties_cleaned.csv ({len(df_prop_clean)} 行)")

if __name__ == "__main__":
    clean_titanium_data()
