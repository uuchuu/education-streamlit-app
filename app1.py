# 导入所需库
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = [
    'PingFang SC',
    'SimHei',
    'Noto Sans CJK SC'
]
plt.rcParams['axes.unicode_minus'] = False


# 设置页面标题和布局
st.set_page_config(
    page_title="中国教育招生数据分析系统",
    page_icon="🎓",
    layout="wide"
)

# 页面标题
st.title("🎓 中国各级各类教育招生数据分析系统")
st.markdown("**基于1978-2024年《中国统计年鉴》教育招生数据**")


@st.cache_data
def load_data():
    """从处理后的CSV文件加载数据"""
    try:
        # 读取CSV文件
        df = pd.read_csv('处理后的数据.csv', encoding='utf-8')

        # 如果CSV文件有BOM头（如你的文件开头有\ufeff），需要处理
        df.columns = df.columns.str.strip()

        # 列名映射：将CSV列名映射到原代码中的列名
        column_mapping = {
            '年份': '年份',
            '研究生': '研究生',
            '普通、职业本专科': '普通本专科',
            '#专科': '成人本专科',
            '普通高中': '普通高中',
            '中等职业教育': '中等职业教育',
            '初中阶段': '初中阶段',
            '小学阶段': '小学阶段',
            '特殊教育': '特殊教育',
            '学前教育': '学前教育'
        }

        # 只保留需要的列并重命名
        df = df[list(column_mapping.keys())].copy()
        df = df.rename(columns=column_mapping)

        # 数据类型转换
        df['年份'] = df['年份'].astype(int)

        # 检查是否有必要的列缺失
        required_columns = ['年份', '研究生', '普通本专科', '成人本专科', '普通高中',
                            '中等职业教育', '初中阶段', '小学阶段', '特殊教育', '学前教育']

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.warning(f"数据文件中缺少以下列: {missing_cols}")

        # 按年份排序
        df = df.sort_values('年份').reset_index(drop=True)

        # 显示加载成功信息
        st.sidebar.success(f"✅ 成功加载 {len(df)} 行数据")

        return df

    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        # 如果CSV文件有问题，提供一个空的DataFrame作为后备
        return pd.DataFrame({
            "年份": [],
            "研究生": [],
            "普通本专科": [],
            "成人本专科": [],
            "普通高中": [],
            "中等职业教育": [],
            "初中阶段": [],
            "小学阶段": [],
            "特殊教育": [],
            "学前教育": []
        })

# 加载数据
df = load_data()

# 侧边栏 - 控制面板
st.sidebar.header("⚙️ 控制面板")

# 显示数据信息
st.sidebar.info(f"**数据范围**: {df['年份'].min()}年 - {df['年份'].max()}年")
st.sidebar.info(f"**教育类型数**: {len(df.columns) - 1}")
st.sidebar.info(f"**总数据量**: {df.shape[0]}个年份 × {df.shape[1]}个指标")

# 主页面 - 数据概览
st.header("📋 数据概览")

# 显示原始数据
with st.expander("查看完整数据表", expanded=False):
    st.dataframe(df.style.format("{:.1f}"), use_container_width=True)

# 显示数据统计信息
with st.expander("查看数据统计摘要", expanded=False):
    st.write(df.describe())

# 1. 单教育类型趋势分析
st.header("📈 单教育类型招生趋势分析")

col1, col2 = st.columns([1, 3])

with col1:
    edu_type = st.selectbox(
        "选择教育类型",
        df.columns[1:],
        index=1,  # 默认选择"普通本专科"
        help="选择要分析的教育类型"
    )

    # 显示选中教育类型的基本统计
    st.metric("最新年份", f"{df['年份'].max()}年")
    st.metric("最新招生人数", f"{df[edu_type].iloc[-1]:.1f} 万人")
    st.metric("历史最高", f"{df[edu_type].max():.1f} 万人")
    st.metric("average increasing rate", f"{((df[edu_type].iloc[-1] / df[edu_type].iloc[0]) ** (1 / len(df)) - 1) * 100:.2f}%")

with col2:
    # 创建趋势图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制折线图
    ax.plot(df["年份"], df[edu_type],
            marker='o',
            linewidth=2.5,
            markersize=6,
            color='#2E86AB',
            label=edu_type)

    # 填充区域
    ax.fill_between(df["年份"], df[edu_type], alpha=0.2, color='#2E86AB')

    ax.set_xlabel("year", fontsize=12, fontweight='bold')
    ax.set_ylabel("Student Enrollment(in ten thousands)", fontsize=12, fontweight='bold')
    ax.set_title(f"Analysis of Enrollment Trends(1978-2024)", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left')

    # 添加数据标签（每隔几年显示一次）
    for i in range(0, len(df), 5):
        ax.text(df["年份"].iloc[i], df[edu_type].iloc[i],
                f'{df[edu_type].iloc[i]:.0f}',
                fontsize=8, ha='center', va='bottom')

    st.pyplot(fig)

# 2. 多教育类型对比分析
st.header("📊 多教育类型对比分析")

selected_types = st.multiselect(
    "选择多个教育类型进行对比",
    df.columns[1:],
    default=["普通本专科", "普通高中", "中等职业教育", "研究生"],
    help="最多选择6个教育类型进行对比"
)

if selected_types:
    # 限制最多选择6个
    selected_types = selected_types[:6]

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # 定义颜色
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E885B', '#6A4C93']

    # 绘制多条折线
    for i, edu in enumerate(selected_types):
        ax2.plot(df["年份"], df[edu],
                 marker='o',
                 linewidth=2,
                 markersize=4,
                 color=colors[i % len(colors)],
                 label=edu)

    ax2.set_xlabel("year", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Student Enrollment(in ten thousands)", fontsize=12, fontweight='bold')
    ax2.set_title("Comparison of Trends in Various Types of Enrollment", fontsize=16, fontweight='bold', pad=20)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 设置y轴为科学计数法（如果数值过大）
    if df[selected_types].max().max() > 1000:
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    st.pyplot(fig2)

    # 显示对比统计表
    with st.expander("查看对比统计表"):
        comparison_df = pd.DataFrame({
            '教育类型': selected_types,
            '2024年招生数': [df[edu].iloc[-1] for edu in selected_types],
            '1978年招生数': [df[edu].iloc[0] if edu in df.columns else 'N/A' for edu in selected_types],
            '增长倍数': [(df[edu].iloc[-1] / df[edu].iloc[0]) if edu in df.columns else 'N/A' for edu in selected_types]
        })
        st.dataframe(comparison_df)

# 3. 最新年份数据对比（柱状图）
st.header("🏆 2024年各教育类型招生人数对比")

# 获取2024年数据
latest_year = df.iloc[-1]

# 创建水平柱状图
fig3, ax3 = plt.subplots(figsize=(12, 8))

# 按数值排序
edu_names = df.columns[1:]
values = latest_year[1:].values
sorted_indices = np.argsort(values)
sorted_edu_names = [edu_names[i] for i in sorted_indices]
sorted_values = values[sorted_indices]

# 创建渐变色
colors = plt.cm.Blues(np.linspace(0.4, 1, len(edu_names)))

bars = ax3.barh(sorted_edu_names, sorted_values, color=colors, edgecolor='black', height=0.7)

ax3.set_xlabel("Student Enrollment(in ten thousands)", fontsize=12, fontweight='bold')
ax3.set_title("Comparison of Enrollment Numbers for Various Education Types in 2024", fontsize=16, fontweight='bold', pad=20)
ax3.grid(True, axis='x', linestyle='--', alpha=0.3)

# 在柱状图上显示数值
for bar, value in zip(bars, sorted_values):
    width = bar.get_width()
    ax3.text(width + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
             f'{value:.1f}', va='center', fontsize=10)

st.pyplot(fig3)

# 4. 机器学习预测模块
st.header("🔮 普通本专科招生人数预测")

# 训练线性回归模型
X = df[["年份"]].values
y = df["普通本专科"].values

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 预测功能界面
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    input_year = st.number_input(
        "请输入预测年份",
        min_value=2025,
        max_value=2050,
        value=2030,
        step=1
    )

with col2:
    st.write("")
    st.write("")
    predict_btn = st.button("🔮 开始预测", type="primary", use_container_width=True)

with col3:
    if predict_btn:
        pred_value = model.predict([[input_year]])[0]
        st.success(f"### {input_year}年预测值")
        st.success(f"## {pred_value:.1f} 万人")

# 显示预测趋势图
fig4, ax4 = plt.subplots(figsize=(12, 6))

# 绘制历史数据
ax4.scatter(df["年份"], df["普通本专科"],
            color='blue',
            label="历史实际数据",
            s=60,
            alpha=0.7,
            edgecolors='black')

# 绘制回归线
years_extended = np.arange(1978, 2031).reshape(-1, 1)
predictions_extended = model.predict(years_extended)
ax4.plot(years_extended, predictions_extended,
         color='red',
         linewidth=3,
         label="线性回归预测线",
         linestyle='--',
         alpha=0.8)

# 标记未来预测点
if predict_btn:
    ax4.scatter([input_year], [pred_value],
                color='green',
                s=200,
                marker='*',
                label=f"{input_year}年预测点",
                edgecolors='black',
                linewidth=2)

ax4.set_xlabel("year", fontsize=12, fontweight='bold')
ax4.set_ylabel("Student Enrollment(in ten thousands)", fontsize=12, fontweight='bold')
ax4.set_title("Trends in Ordinary Undergraduate and Associate Degree Enrollment and Linear Regression Prediction", fontsize=16, fontweight='bold', pad=20)
ax4.legend(loc='upper left')
ax4.grid(True, linestyle='--', alpha=0.3)

st.pyplot(fig4)

# 模型评估指标
st.subheader("📊 模型评估指标")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("均方误差 (MSE)", f"{mse:.2f}")

with col2:
    st.metric("R² 决定系数", f"{r2:.4f}")

with col3:
    st.metric("模型斜率", f"{model.coef_[0]:.2f}")

# 5. 数据下载功能
st.header("📥 数据导出")

# 转换为CSV
csv = df.to_csv(index=False, encoding='utf-8-sig')

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📄 下载完整数据 (CSV)",
        data=csv,
        file_name="中国教育招生数据_1978-2024.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    # 生成项目报告摘要
    report = f"""中国教育招生数据分析报告
数据范围: {df['年份'].min()}年-{df['年份'].max()}年
教育类型数量: {len(df.columns) - 1}

主要发现:
1. 普通本专科招生增长显著: {df['普通本专科'].iloc[-1]:.1f}万人 (2024年)
2. 研究生教育持续增长: {df['研究生'].iloc[-1]:.1f}万人 (2024年)
3. 小学阶段招生呈下降趋势

预测模型性能:
- R²决定系数: {r2:.4f}
- 均方误差: {mse:.2f}

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    st.download_button(
        label="📝 下载分析报告 (TXT)",
        data=report,
        file_name="教育招生数据分析报告.txt",
        mime="text/plain",
        use_container_width=True
    )

# 页脚信息
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("👥 **项目组成员**")
    st.caption("- 蒋雨彤: 数据处理与模型构建")
    st.caption("- 丁楚钧: 可视化与系统部署")

with footer_col2:
    st.caption("📚 **数据来源**")
    st.caption("《中国统计年鉴2025》")
    st.caption("国家统计局")

with footer_col3:
    st.caption("🛠️ **技术栈**")
    st.caption("Python • Streamlit • Pandas")
    st.caption("Scikit-learn • Matplotlib")

st.caption("© 2025 数据处理与可视化课程项目 • 上海师范大学")