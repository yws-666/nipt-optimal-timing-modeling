import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Tuple, List, Dict, Optional
import os


# ----------------- 配置项 -----------------
# --- 路径配置：按项目目录结构管理输入输出 ---
_BASE = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_BASE, '..', '数据与题目', '附件.xlsx')
_FIG_DIR = os.path.join(_BASE, '..', '图片', '问题三')
_RESULT_DIR = os.path.join(_BASE, '..', '结果文件')

class Config:
    KEY_COLUMNS = ['孕妇代码', '检测日期', '孕周_小数', '孕妇BMI', '年龄', '身高', '体重','Y染色体浓度', '重复读段的比例', '被过滤掉读段数的比例', 'GC含量','原始读段数', '在参考基因组上比对的比例',
        '唯一比对的读段数  ', '检测抽血次数']
    QUALIFIED_Y_THRESHOLD=0.04
    GC_NORMAL_RANGE=(0.4, 0.6)
    VIF_THRESHOLD=5  # VIF阈值

    BMI_CLUSTER_MAX_K=6
    DEFAULT_BEST_K =3
    # 最佳时点
    MIN_GESTATIONAL_WEEK =10
    MAX_GESTATIONAL_WEEK= 25
    WEEK_STEP =0.5
    QUALIFIED_RATIO_THRESHOLD =0.9
    SUBGROUP_QUALIFIED_THRESHOLD= 0.85
    MIN_SUBGROUP_SIZE= 5

    FIG_DPI = 300
    RESULT_FILE=os.path.join(_RESULT_DIR, 'NIPT问题三结果汇总.xlsx')
    CORR_HEATMAP=os.path.join(_FIG_DIR, '相关性热力图_优化.png')
    BMI_ELBOW_PLOT=os.path.join(_FIG_DIR, 'BMI聚类肘部图.png')
    BMI_SCATTER_PLOT=os.path.join(_FIG_DIR, 'BMI与达标时间散点图.png')  # 新增图表
    BMI_BOXPLOT=os.path.join(_FIG_DIR, 'BMI分组箱线图.png')
    BEST_TIME_RISK_BAR=os.path.join(_FIG_DIR, '最佳时间与风险条形图.png')  # 新增图表
    ERROR_BOXPLOT=os.path.join(_FIG_DIR, '误差分组达标时间箱线图.png')
    ERROR_IMPACT_PLOT=os.path.join(_FIG_DIR, '误差对最佳时点的影响.png')
    VIF_PLOT=os.path.join(_FIG_DIR, '变量方差膨胀因子.png')
    PCA_VARIANCE=os.path.join(_FIG_DIR, '主成分方差解释率.png')


def init():

    os.makedirs(_FIG_DIR, exist_ok=True)
    os.makedirs(_RESULT_DIR, exist_ok=True)
    warnings.filterwarnings('ignore')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

def save(fig: plt.Figure, filename: str, title: Optional[str] = None) -> None:
    """保存图表到当前目录"""
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(filename, dpi=Config.FIG_DPI)
    plt.close(fig)


def plot_vif(vif_df: pd.DataFrame, title: str = "变量方差膨胀因子(VIF)") -> plt.Figure:
    """绘制VIF值柱状图"""
    vif_df = vif_df.sort_values("VIF", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(vif_df["特征"], vif_df["VIF"], color='skyblue')
    ax.axhline(y=Config.VIF_THRESHOLD, color='r', linestyle='--', label=f'VIF阈值={Config.VIF_THRESHOLD}')
    ax.set_xlabel('变量')
    ax.set_ylabel('VIF值')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    return fig


def plot_pca_variance(pca: PCA) -> plt.Figure:#绘制主成分方差解释率
    fig, ax = plt.subplots(figsize=(10, 6))
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    components = range(1, len(explained_variance) + 1)
    ax.bar(components, explained_variance, alpha=0.7, label='单个主成分解释率')
    ax.plot(components, cumulative_variance, 'r-o', label='累积方差解释率')
    ax.axhline(y=0.9, color='g', linestyle='--', label='90%方差解释率阈值')
    ax.set_xlabel('主成分序号')
    ax.set_ylabel('方差解释率')
    ax.set_title('主成分分析(PCA)方差解释率')
    ax.set_xticks(components)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_bmi_vs_time(df_pregnant: pd.DataFrame) -> plt.Figure:#绘制BMI与达标时间的散点图
    fig, ax = plt.subplots(figsize=(10, 7))
    groups = df_pregnant['BMI分组'].unique()
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'orchid', 'gold', 'teal']
    for i, group in enumerate(groups):
        group_data = df_pregnant[df_pregnant['BMI分组'] == group]
        ax.scatter(group_data['孕妇BMI'], group_data['达标时间'],
                   label=f'BMI分组 {group}', color=colors[i], alpha=0.6, s=30)

    z = np.polyfit(df_pregnant['孕妇BMI'], df_pregnant['达标时间'], 1)
    p = np.poly1d(z)
    ax.plot(df_pregnant['孕妇BMI'], p(df_pregnant['孕妇BMI']), 'k--', linewidth=1.5)

    ax.set_xlabel('孕妇BMI')
    ax.set_ylabel('达标时间 (周)')
    ax.set_title('BMI与达标时间的关系散点图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_bmi_group_boxplot(df_pregnant: pd.DataFrame) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(10, 7))
    df_pregnant.boxplot(column='孕妇BMI',by='BMI分组',grid=False, ax=ax)
    ax.set_xlabel('BMI分组')
    ax.set_ylabel('BMI值')
    ax.set_title('各BMI分组的BMI分布箱线图')
    plt.suptitle('')  # 去除默认标题
    return fig
#绘制不同BMI误差子组箱线图
def plot_error_subgroup_boxplot(df_pregnant: pd.DataFrame) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(12, 8))
    df_pregnant.boxplot(column='达标时间', by=['BMI分组', '误差分组'], grid=False, ax=ax)
    ax.set_xlabel('BMI分组-误差分组')
    ax.set_ylabel('达标时间 (周)')
    ax.set_title('不同BMI分组和误差分组的达标时间分布箱线图')
    plt.suptitle('')
    plt.xticks(rotation=45,ha='right')
    return fig

def plot_best_time_risk_bar(best_timepoints_df: pd.DataFrame) -> plt.Figure:  #绘制最佳时间与风险条形图

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(best_timepoints_df['BMI分组'], best_timepoints_df['最佳时点'], color='skyblue', zorder=3)
    ax1.set_xlabel('BMI分组')
    ax1.set_ylabel('最佳NIPT时点 (周)')
    ax1.set_title('各BMI分组最佳NIPT时点')
    ax1.set_xticks(best_timepoints_df['BMI分组'])
    for i, v in enumerate(best_timepoints_df['最佳时点']):
        ax1.text(i,v + 0.1, f'{v:.1f}', ha='center', fontsize=9)

    ax2.bar(best_timepoints_df['BMI分组'], best_timepoints_df['平均风险'], color='lightcoral', zorder=3)
    ax2.set_xlabel('BMI分组')
    ax2.set_ylabel('平均风险')
    ax2.set_title('各BMI分组平均风险')
    ax2.set_xticks(best_timepoints_df['BMI分组'])
    ax2.set_ylim(0, 0.5)
    for i, v in enumerate(best_timepoints_df['平均风险']):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    # 添加网格线
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    return fig


def yunzhou(week_str: str) -> float:
    if pd.isna(week_str):return np.nan
    match=re.match(r'(\d+)w(?:\+(\d+))?', str(week_str))
    if not match:return np.nan
    weeks, days=match.groups()
    return int(weeks)+(int(days) if days else 0)/7

def yun_fu_zui_zao_da_biao(group: pd.DataFrame) -> Optional[float]:
    """获取每个孕妇最早达标时间"""
    qualified = group[group['Y染色体浓度'] >= Config.QUALIFIED_Y_THRESHOLD]
    return qualified['孕周_小数'].min() if not qualified.empty else np.nan


def data(file_path: str) -> pd.DataFrame:
    """加载并预处理数据，返回每个孕妇的唯一记录"""
    try:
        df=pd.read_excel(file_path,sheet_name="男胎检测数据")
        print(f"原始数据总行数: {len(df)}")
    except Exception as e:
        raise ValueError(f"读取文件'{file_path}'失败: {e}")

    df_male=df[df['Y染色体浓度'].notna()].copy()
    print(f"筛选出男胎数据条数：{len(df_male)}")

    df_male['孕周_小数']=df_male['检测孕周'].apply(yunzhou)
    df_male.dropna(subset=['孕周_小数'], inplace=True)

    df_male['检测日期']=pd.to_datetime(df_male['检测日期'],errors='coerce')
    df_male.dropna(subset=['检测日期'],inplace=True)
    df_male.sort_values(['孕妇代码', '检测日期'], inplace=True)

    qualified_time_map=df_male.groupby('孕妇代码').apply(yun_fu_zui_zao_da_biao)
    df_male['达标时间']=df_male['孕妇代码'].map(qualified_time_map)

    df_qualified=df_male.dropna(subset=['达标时间']).copy()
    print(f"有达标时间的孕妇数据条数：{len(df_qualified)}")

    df_qualified['GC异常'] = ((df_qualified['GC含量'] < Config.GC_NORMAL_RANGE[0]) |(df_qualified['GC含量'] > Config.GC_NORMAL_RANGE[1])).astype(int)
    df_qualified['误差评分'] = (df_qualified['重复读段的比例'] +df_qualified['被过滤掉读段数的比例'] +(1 - df_qualified['在参考基因组上比对的比例']) +df_qualified['GC异常'] * 0.1)

    # 去重
    df_pregnant = df_qualified.loc[df_qualified.groupby('孕妇代码')['检测日期'].idxmax()]
    df_pregnant = df_pregnant.drop_duplicates('孕妇代码').reset_index(drop=True)
    print(f"去重后孕妇数量：{len(df_pregnant)}")

    bmi_anomalies = df_pregnant[(df_pregnant['孕妇BMI'] < 15) | (df_pregnant['孕妇BMI'] > 40)].shape[0]
    if bmi_anomalies > 0:
        print(f"检测到{bmi_anomalies}条异常BMI数据（<15或>40）")

    return df_pregnant

def analyze_influencing_factors(df_pregnant: pd.DataFrame) -> None:
    """分析影响因素，处理多重共线性并建立回归模型"""
    factors=['孕妇BMI', '年龄', '身高', '体重', 'Y染色体浓度', '重复读段的比例','被过滤掉读段数的比例', 'GC含量', '原始读段数', '在参考基因组上比对的比例','唯一比对的读段数  ', '误差评分']
    target='达标时间'

    corr_matrix = df_pregnant[factors + [target]].corr()
    print("\n相关性矩阵")
    print(corr_matrix[target].sort_values(ascending=False))

    # VIF计算
    X=df_pregnant[factors].copy().dropna()
    y=df_pregnant.loc[X.index, target]

    vif_data=pd.DataFrame()
    vif_data["特征"]=X.columns
    vif_data["VIF"]=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVIF")
    print(vif_data.sort_values("VIF", ascending=False))
    save(plot_vif(vif_data), Config.VIF_PLOT)

    high_vif_vars= vif_data[vif_data["VIF"] > Config.VIF_THRESHOLD]["特征"].tolist()
    low_vif_vars =vif_data[vif_data["VIF"] <= Config.VIF_THRESHOLD]["特征"].tolist()
    X_processed =X[low_vif_vars].copy()
    if high_vif_vars:
        print(f"\n检测到高共线性变量: {high_vif_vars}，进行主成分分析降维...")

        scaler = StandardScaler()
        X_high_vif_scaled = scaler.fit_transform(X[high_vif_vars])

        pca = PCA(n_components=0.9)  # 保留90%的方差
        pca_result = pca.fit_transform(X_high_vif_scaled)

        print(f"PCA保留了 {pca.n_components_} 个主成分, 解释了{sum(pca.explained_variance_ratio_):.2%} 的方差。")
        save(plot_pca_variance(pca), Config.PCA_VARIANCE)

        for i in range(pca.n_components_):
            X_processed[f'PCA_{i + 1}']=pca_result[:, i]

    model = LinearRegression()
    n_features=max(1, X_processed.shape[1] // 3)
    selector=RFE(model, n_features_to_select=n_features, step=1)
    selector.fit(X_processed, y)

    selected_features=X_processed.columns[selector.support_]
    print(f"\n逐步回归选中的变量: {selected_features.tolist()}")

    # 使用statsmodels拟合最终模型
    X_final = sm.add_constant(X_processed[selected_features])
    final_model = sm.OLS(y, X_final).fit()

    print("\n=== 多元线性回归结果 ===")
    print(final_model.summary())


def find_optimal_k(inertias: List[float], k_range: range) -> int:#肘部法
    if len(inertias) < 2:
        return Config.DEFAULT_BEST_K

    diffs=np.diff(inertias)
    diff_ratios = np.diff(diffs)

    optimal_idx = np.argmax(abs(diff_ratios)) + 1  # +1 补偿diff导致的索引偏移
    return k_range[optimal_idx] if optimal_idx < len(k_range) else Config.DEFAULT_BEST_K

def bmi_clustering(df_pregnant: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:#K-means聚类
    X_bmi= df_pregnant[['孕妇BMI']].values
    scaler= StandardScaler()
    X_bmi_scaled=scaler.fit_transform(X_bmi)

    inertias=[]
    k_range=range(2, Config.BMI_CLUSTER_MAX_K + 1)
    for k in k_range:
        kmeans =KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_bmi_scaled)
        inertias.append(kmeans.inertia_)

    best_k=find_optimal_k(inertias, k_range)
    print(f"通过肘部法则确定的最佳BMI聚类数量: {best_k}")

    fig,ax=plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, 'bo-')
    ax.axvline(x=best_k, color='r', linestyle='--', label=f'最佳K={best_k}')
    ax.set_xlabel('聚类数量K')
    ax.set_ylabel('簇内平方和（Inertia）')
    ax.set_title('BMI聚类肘部法则图')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig,Config.BMI_ELBOW_PLOT)

    kmeans=KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_pregnant['BMI分组']=kmeans.fit_predict(X_bmi_scaled)

    group_bmi_mean=df_pregnant.groupby('BMI分组')['孕妇BMI'].mean().sort_values()
    group_mapping={old: new for new, old in enumerate(group_bmi_mean.index)}
    df_pregnant['BMI分组']=df_pregnant['BMI分组'].map(group_mapping)

    group_stats=df_pregnant.groupby('BMI分组').agg({
        '孕妇BMI':['count','min','mean','max','std'],
        '达标时间':['mean','std']
    }).round(4)
    group_stats.columns=['孕妇数量', 'BMI最小值', 'BMI平均值', 'BMI最大值', 'BMI标准差','平均达标时间', '达标时间标准差']
    print("\nBMI分组结果")
    print(group_stats)

    # BMI与达标时间散点图
    scatter_fig = plot_bmi_vs_time(df_pregnant)
    save(scatter_fig, Config.BMI_SCATTER_PLOT)

    # BMI分组箱线图
    boxplot_fig = plot_bmi_group_boxplot(df_pregnant)
    save(boxplot_fig, Config.BMI_BOXPLOT)

    return df_pregnant, group_stats


def calculate_risk(gestational_week: float) -> float:
    if gestational_week <= 12: return 0.2
    if 12 < gestational_week <= 27: return 0.2 + (gestational_week - 12) * 0.02
    return 1.0


def find_best_timepoint_for_group(group_data, threshold, min_w, max_w, step) -> Optional[pd.DataFrame]:#找到最佳检测时点
    if len(group_data) == 0: return None
    timepoints =np.arange(min_w, max_w + step, step)
    results =[]
    for tp in timepoints:
        ratio= (group_data['达标时间'] <= tp).sum() / len(group_data)
        if ratio>=threshold:
            results.append({'时点': tp,'达标比例': ratio,'平均风险':calculate_risk(tp)})
    return pd.DataFrame(results) if results else None


def find_best_timepoint(df_pregnant: pd.DataFrame) -> pd.DataFrame:
    #找到最佳NIPT检测时点
    best_timepoints = []
    for group in sorted(df_pregnant['BMI分组'].unique()):
        group_data=df_pregnant[df_pregnant['BMI分组']==group]
        results_df=find_best_timepoint_for_group(group_data, Config.QUALIFIED_RATIO_THRESHOLD, Config.MIN_GESTATIONAL_WEEK,Config.MAX_GESTATIONAL_WEEK, Config.WEEK_STEP)
        if results_df is not None and not results_df.empty:
            best_tp=results_df.loc[results_df['平均风险'].idxmin()]
            best_timepoints.append({'BMI分组': group, '最佳时点': best_tp['时点'], '达标比例': best_tp['达标比例'],'平均风险': best_tp['平均风险'], '孕妇数量': len(group_data)})
    best_df = pd.DataFrame(best_timepoints)
    print("各BMI分组最佳NIPT时点")
    print(best_df.round(4))

    bar_fig=plot_best_time_risk_bar(best_df)
    save(bar_fig,Config.BEST_TIME_RISK_BAR)

    return best_df

def analyze_error_impact(df_pregnant: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:#分析检测误差对达标时间和最佳时点的影响
    df_pregnant['误差分组'] = pd.qcut(df_pregnant['误差评分'], 3, labels=['低误差','中误差','高误差'])
    error_stats = df_pregnant.groupby(['BMI分组','误差分组'])['达标时间'].agg(['count','mean','std']).round(4)
    error_stats.columns = ['孕妇数量','平均达标时间','达标时间标准差']
    print("误差分组对达标时间的影响")
    print(error_stats)

    # 误差子组箱线图
    error_boxplot_fig=plot_error_subgroup_boxplot(df_pregnant)
    save(error_boxplot_fig,Config.ERROR_BOXPLOT)

    sub_best_timepoints=[]
    for bmi_group in sorted(df_pregnant['BMI分组'].unique()):
        for error_group in ['低误差','中误差','高误差']:
            sub_data = df_pregnant[(df_pregnant['BMI分组'] == bmi_group) & (df_pregnant['误差分组'] == error_group)]
            if len(sub_data)<Config.MIN_SUBGROUP_SIZE: continue

            results_df=find_best_timepoint_for_group(sub_data,Config.SUBGROUP_QUALIFIED_THRESHOLD,Config.MIN_GESTATIONAL_WEEK,Config.MAX_GESTATIONAL_WEEK, Config.WEEK_STEP)

            if results_df is not None and not results_df.empty:
                best_tp = results_df.loc[results_df['平均风险'].idxmin()]
                sub_best_timepoints.append({'BMI分组':bmi_group,'误差分组':error_group,'时点':best_tp['时点'],
                    '达标比例': best_tp['达标比例'],'平均风险':best_tp['平均风险']})

    sub_best_df = pd.DataFrame(sub_best_timepoints)
    print("\n各BMI-误差亚组最佳时点 ")
    print(sub_best_df.round(4))
    return error_stats, sub_best_df


def summarize_results(bmi_group_stats: pd.DataFrame,best_timepoints_df: pd.DataFrame,sub_best_df: pd.DataFrame,error_stats: pd.DataFrame):#汇总到Excel

    bmi_ranges = bmi_group_stats.apply(lambda row: f"[{row['BMI最小值']:.1f},{row['BMI最大值']:.1f}]", axis=1).rename('BMI区间')
    summary_df = pd.concat([bmi_group_stats, bmi_ranges], axis=1).reset_index()
    summary_df.rename(columns={'BMI平均值': '平均BMI'}, inplace=True)
    final_result = pd.merge(summary_df,best_timepoints_df.drop(columns=['孕妇数量']),on='BMI分组')

    output_columns = ['BMI分组','BMI区间','平均BMI', '孕妇数量','最佳时点','达标比例', '平均风险']
    final_output = final_result[output_columns]

    print("\n最终结果汇总 ")
    print(final_output.round(4))

    try:
        with pd.ExcelWriter(Config.RESULT_FILE,engine='openpyxl') as writer:
            final_output.to_excel(writer,sheet_name='BMI分组与最佳时点',index=False)
            sub_best_df.to_excel(writer,sheet_name='误差亚组最佳时点',index=False)
            bmi_group_stats.to_excel(writer,sheet_name='BMI分组统计')
            error_stats.to_excel(writer,sheet_name='误差对达标时间影响')
        print(f"结果已保存到: {Config.RESULT_FILE}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")


# ----------------- 主函数 -----------------
def main(file_path: str = "附件.xlsx"):
    init()
    df_pregnant=data(file_path)
    analyze_influencing_factors(df_pregnant.copy())
    df_pregnant,bmi_group_stats=bmi_clustering(df_pregnant.copy())
    best_timepoints_df=find_best_timepoint(df_pregnant.copy())
    error_stats,sub_best_df=analyze_error_impact(df_pregnant.copy())
    summarize_results(bmi_group_stats, best_timepoints_df, sub_best_df, error_stats)

if __name__ == "__main__":
    main(file_path=_DATA_PATH)