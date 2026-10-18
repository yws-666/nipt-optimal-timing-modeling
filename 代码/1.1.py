import pandas as pd
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Gamma
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import re
# --- 路径配置：按项目目录结构管理输入输出 ---
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE, '..', '数据与题目', '附件.xlsx')
FIG_DIR = os.path.join(_BASE, '..', '图片', '问题1')
RESULT_DIR = os.path.join(_BASE, '..', '结果文件')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

warnings.filterwarnings('ignore', category=UserWarning)

plt.rcParams['font.sans-serif']=['Microsoft YaHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['mathtext.fontset']='dejavusans'
sns.set_style("whitegrid",{'font.sans-serif':['Microsoft YaHei', 'DejaVu Sans']})

# 孕周转换
def yun_zhou_zhuan_huan(age_str):
    if pd.isna(age_str):return np.nan
    age_str = str(age_str).strip().lower()
    patterns = [r'^(\d+)\s*[w周]\s*\+?\s*(\d+)\s*天?$',r'^(\d+)\.?\d?\s*[w周]\s*\d+\.?\d?$' ]
    for pattern in patterns:
        match=re.match(pattern, age_str)
        if match:
            weeks=int(match.group(1))
            days=int(match.group(2)) if match.group(2) else 0
            return weeks+days/7
    try:
        return float(age_str)
    except ValueError:
        return np.nan
# 预处理
df=pd.read_excel(DATA_PATH, sheet_name='男胎检测数据', engine='openpyxl')
core_cols=['孕妇代码','检测孕周','孕妇BMI','Y染色体浓度','GC含量','在参考基因组上比对的比例','被过滤掉读段数的比例','原始读段数','重复读段的比例','唯一比对的读段数  ','胎儿是否健康']
df_core=df[core_cols].dropna(subset=core_cols).copy()
df_core['孕周_数值']=df_core['检测孕周'].apply(yun_zhou_zhuan_huan)
df_core=df_core.dropna(subset=['孕周_数值'])

df_clean=df_core[
    (df_core['Y染色体浓度']>0)&(df_core['Y染色体浓度']<0.2)&(df_core['孕妇BMI']>=18) & (df_core['孕妇BMI']<= 50) &
    (df_core['孕周_数值']>=10)&(df_core['孕周_数值']<=25)&(df_core['GC含量']>=0.3) & (df_core['GC含量']<= 0.7) &
    (df_core['在参考基因组上比对的比例']<=1)&(df_core['被过滤掉读段数的比例']<=0.5)&(df_core['原始读段数'] >=10000) &
    (df_core['重复读段的比例']<=0.8)&(df_core['唯一比对的读段数  ']>=5000)].copy()

numeric_cols =['Y染色体浓度','GC含量', '在参考基因组上比对的比例','被过滤掉读段数的比例','原始读段数','孕妇BMI','孕周_数值','重复读段的比例', '唯一比对的读段数  ']
for col in numeric_cols:
    if df_clean[col].dtype=='object':
        df_clean[col] =pd.to_numeric(df_clean[col], errors='coerce')
df_clean =df_clean.dropna(subset=numeric_cols)

df_clean['胎儿健康_二值化'] = df_clean['胎儿是否健康'].map({'是': 1, '否': 0})

y = df_clean['Y染色体浓度'].astype(float)
gestation_week = df_clean['孕周_数值'].astype(float)
bmi = df_clean['孕妇BMI'].astype(float)

# 输出预处理结果
print(f"原始数据行数：{len(df)}")
print(f"预处理后有效行数：{len(df_clean)}")

print('\n')
print("Y染色体浓度与各指标的相关性\n")
analysis_vars = [
    (gestation_week,'孕周'),(bmi,'BMI'),(df_clean['GC含量'],'GC含量'),(df_clean['在参考基因组上比对的比例'],'参考基因组比对比例'),
    (df_clean['被过滤掉读段数的比例'],'过滤读段比例'),(np.log(df_clean['原始读段数']), 'log(原始读段数)'),(df_clean['重复读段的比例'], '重复读段比例'),
(df_clean['唯一比对的读段数  '], '唯一比对读段数'),]

# 计算并输出相关系数
corr_results=[]
for var,name in analysis_vars:
    corr, p = stats.pearsonr(var, y)
    corr_results.append((name, corr, p, '显著相关' if p < 0.05 else '不显著相关'))

corr_df = pd.DataFrame(corr_results, columns=['指标','皮尔逊相关系数','p值','显著性'])
for i in range(len(corr_df)):
    row = corr_df.iloc[i]
    print(f"- {row['指标']}：相关系数={row['皮尔逊相关系数']:.6f}，p值={row['p值']:.4f}，{row['显著性']}")

print(f"\n显著相关指标数量：{len(corr_df[corr_df['p值'] < 0.05])}个")

# 相关性热力图
plt.figure(figsize=(10, 8))
corr_matrix = pd.DataFrame({
    '孕周': gestation_week,'BMI': bmi,'GC含量': df_clean['GC含量'],
    '参考基因组比对比例':df_clean['在参考基因组上比对的比例'],
    '过滤读段比例':df_clean['被过滤掉读段数的比例'],
    'log(原始读段数)':np.log(df_clean['原始读段数']),
    '重复读段比例':df_clean['重复读段的比例'],
    '唯一比对读段数':df_clean['唯一比对的读段数  '],
    'Y染色体浓度':y
}).corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix,mask=mask,annot=True,fmt=".2f", cmap="coolwarm",
            cbar_kws={"shrink": .8}, linewidths=0.5,vmin=-1,vmax=1)
plt.title('Y染色体浓度与各指标相关性热力图',fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '图0_相关性热力图.png'), dpi=300)
plt.close()

# 回归模型构建
print("\n")
print("Y染色体浓度多元回归模型\n")

# 中心化处理
bmi_centered=bmi-bmi.mean()
gestation_centered=gestation_week-gestation_week.mean()

# 优化模型
X_optimized=pd.DataFrame({
    '孕周(中心化)':gestation_centered, 'BMI(中心化)': bmi_centered, 'BMI二次项': bmi_centered **2,
    '重复读段比例':df_clean['重复读段的比例'],'过滤读段比例': df_clean['被过滤掉读段数的比例']})
# 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_optimized, y)

print(f"R²）:{ridge.score(X_optimized, y):.4f}")
print(f"（调整R²）：{1 - (1 - ridge.score(X_optimized, y)) * (len(y) - 1) / (len(y) - len(X_optimized.columns) - 1):.4f}")

# 回归系数表
coef_table=pd.DataFrame({
    '自变量':X_optimized.columns,'系数值':ridge.coef_.round(6),'标准化系数':(ridge.coef_ * X_optimized.std(axis=0))/ y.std()})
print("\n回归系数表：")
print(coef_table.to_string(index=False))

# 混合效应模型
model_mixed=MixedLM(endog=y, exog=sm.add_constant(X_optimized), groups=df_clean['孕妇代码'])
result_mixed= model_mixed.fit()
print(result_mixed.summary())

# Gamma回归
gamma_model =GLM(y, sm.add_constant(X_optimized), family=Gamma())
gamma_results = gamma_model.fit()
print(gamma_results.summary())

# 模型检验
print("\n")

# 拟合优度检验
print("拟合优度：")
print(f"岭回归模型解释Y染色体浓度变异比例：{ridge.score(X_optimized, y)*100:.2f}%")
print(f"混合效应模型对数似然值：{result_mixed.llf:.4f}")
print(f"Gamma回归模型伪R²:{gamma_results.pseudo_rsquared(kind='mcfadden'):.4f}\n")

# 整体显著性检验
print("整体显著性：")
print(f"混合效应模型似然比：{result_mixed.llf:.2f}")
print(f"Gamma回归模型似然比：{gamma_results.llf:.2f}")

# 多重共线性检验
print("多重共线性（VIF）：")
VIFdata = pd.DataFrame({'自变量': X_optimized.columns,'VIF值': [variance_inflation_factor(X_optimized.values, i) for i in range(X_optimized.shape[1])]})
print(VIFdata.to_string(index=False))
print(f"结论：{'无严重多重共线性' if VIFdata['VIF值'].max() < 5 else '存在严重多重共线性'}")

# 残差检验
residuals=result_mixed.resid
fitted_values=result_mixed.fittedvalues

shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(500, len(residuals))])
print(f"\n残差检验（混合效应模型）：")
print(f"正态性（Shapiro-Wilk）：统计量={shapiro_stat:.4f}，p={shapiro_p:.4f}")
print(f"{'残差近似正态' if shapiro_p > 0.05 else '残差非正态'}")

dw_stat=durbin_watson(residuals)
print(f"独立性（Durbin-Watson）：统计量={dw_stat:.4f}")
print(f"{'残差无自相关' if 1.5 <dw_stat< 2.5 else '残差存在自相关'}")


# 指标影响力排序
X_scaled=pd.DataFrame(StandardScaler().fit_transform(X_optimized), columns=X_optimized.columns)
y_scaled=StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel()
model_scaled = sm.OLS(y_scaled, sm.add_constant(X_scaled)).fit()
scaled_coefs = model_scaled.params[1:]

# 输出影响力排序
impact_df = pd.DataFrame({'自变量':scaled_coefs.index,'标准化系数': scaled_coefs.values,'绝对值': np.abs(scaled_coefs.values)}).sort_values('绝对值', ascending=False)
print("\n指标影响力排行：")
for i, (idx, row) in enumerate(impact_df.iterrows()):
    print(f"{i+1}.{row['自变量']}：标准化系数={row['标准化系数']:.4f}")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.regplot(x=gestation_week,y=y,ax=ax1,scatter_kws={'alpha': 0.6, 'color': '#1f77b4'},line_kws={'color': 'red', 'linestyle': '--'})
ax1.set_title('孕周与Y染色体浓度的关系',fontsize=12)
ax1.set_xlabel('孕周（周）',fontsize=10)
ax1.set_ylabel('Y染色体浓度',fontsize=10)
ax1.grid(True, alpha=0.3)

sns.regplot(x=bmi,y=y,order=2,ax=ax2,scatter_kws={'alpha': 0.6, 'color': '#2ca02c'},line_kws={'color': 'red', 'linestyle': '--'})
ax2.set_title('BMI与Y染色体浓度的非线性关系', fontsize=12)
ax2.set_xlabel('BMI（kg/m²）', fontsize=10)
ax2.set_ylabel('Y染色体浓度', fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '图1_核心指标关系.png'), dpi=300)
plt.close()


plt.figure(figsize=(12, 6))
impact_df_sorted = impact_df.sort_values('绝对值', ascending=True)
bars = plt.barh(impact_df_sorted['自变量'],impact_df_sorted['绝对值'], color='#3292a8')
for i, bar in enumerate(bars):
    plt.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,f'系数={impact_df_sorted["标准化系数"].iloc[i]:.4f}',va='center',fontsize=9)
plt.title('Y染色体浓度影响因素标准化系数',fontsize=14)
plt.xlabel('标准化系数绝对值',fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '图3_指标影响力.png'), dpi=300)
plt.close()


fig4, (ax1, ax2) = plt.subplots(1,2,figsize=(14, 5))
stats.probplot(residuals,plot=ax1)
ax1.set_title('残差Q-Q图（正态性检验）',fontsize=12)
ax1.set_xlabel('理论分位数', fontsize=10)
ax1.set_ylabel('残差分位数',fontsize=10)
ax1.grid(True,alpha=0.3)

sns.regplot(x=fitted_values, y=residuals, lowess=True,scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, ax=ax2)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_title('残差与拟合值关系（同方差性检验）', fontsize=12)
ax2.set_xlabel('拟合值',fontsize=10)
ax2.set_ylabel('残差',fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '图4_残差检验.png'), dpi=300)
plt.close()

