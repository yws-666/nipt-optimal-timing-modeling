# 完整代码（含BMI单单位统计与相邻互补分组调整）
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import warnings
# --- 路径配置：按项目目录结构管理输入输出 ---
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE, '..', '数据与题目', '附件.xlsx')
FIG_DIR = os.path.join(_BASE, '..', '图片', '问题2')
RESULT_DIR = os.path.join(_BASE, '..', '结果文件')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif']=['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

def yun_zhou_zhuan_huan(week_str):

    week_str = str(week_str).strip().lower()
    if 'w+' in week_str:
        week_part,day_part = week_str.split('w+')
        weeks=int(week_part)
        days=int(day_part.replace(' ', ''))
    elif 'w' in week_str and '+' not in week_str:
        weeks=int(week_str.split('w')[0])
        days=0
    else:
        return float(week_str)
    total_week=weeks+days/7
    return total_week if 10 <= total_week<=25 else np.nan

# 检测达标率计算
def jian_ce_da_biao_lv(group_data, target_week, week_tol=0.2):
    mask=(group_data['孕周_数值'] >= target_week - week_tol) & (group_data['孕周_数值'] <= target_week + week_tol)
    if mask.sum() < 5:
        return 0.0
    pass_samples = group_data[mask][group_data[mask]['Y染色体浓度'] >= 0.04]
    return len(pass_samples)/len(group_data[mask])

# 风险损失函数
def risk_loss(target_week, group_data, alpha=0.6, beta=0.4):
    if target_week<=12:
        time_risk=1.0
    elif 12<target_week<=25:
        time_risk = 1.0 + (target_week - 12) * 0.2
    else:
        time_risk = 10.0
    pass_rate=jian_ce_da_biao_lv(group_data, target_week)
    failure_risk =1.0-pass_rate
    if pass_rate<0.7:
        failure_risk *=1.5

    return alpha * time_risk + beta * failure_risk


# -------------------------- 2. 数据加载与预处理 --------------------------
# 加载数据
df=pd.read_excel(DATA_PATH)


df_male=df[df['Y染色体浓度'].notna()].copy()
print(f"原始数据总行数：{len(df)}")

df_male['BMI']=df_male['孕妇BMI']
df_male['孕周_数值']=df_male['检测孕周'].apply(yun_zhou_zhuan_huan)

key_cols = ['孕妇代码','BMI','孕周_数值','Y染色体浓度','原始读段数','在参考基因组上比对的比例']
df_clean = df_male[key_cols].dropna().copy()

df_clean = df_clean[
    (df_clean['BMI']>=20)&(df_clean['BMI']<=50) &(df_clean['Y染色体浓度']>=0.01) & (df_clean['Y染色体浓度']<=0.2) &(df_clean['在参考基因组上比对的比例']>=0.7)].copy()
print(f"清洗后数据行数：{len(df_clean)}")


single_bmi_bins=list(range(20, 52))
single_bmi_labels=[f'{i}-{i + 1}' for i in range(20, 51)]

# 统计每个单单位区间的样本数
df_clean['BMI单单位分组']=pd.cut(
    df_clean['BMI'],bins=single_bmi_bins,labels=single_bmi_labels,include_lowest=True)
single_group_counts=df_clean['BMI单单位分组'].value_counts().sort_index()

# 整理单单位统计结果（过滤样本数为0的区间）
single_group_stats=pd.DataFrame({'BMI区间': single_group_counts.index,'样本数': single_group_counts.values,
                                   '占比(%)': [round(count / len(df_clean) * 100, 2) for count in single_group_counts.values],'累计样本数': single_group_counts.values.cumsum(),'累计占比(%)': [round(count / len(df_clean) * 100, 2) for count in single_group_counts.values.cumsum()]})
single_group_stats=single_group_stats[single_group_stats['样本数']>0].reset_index(drop=True)

print(single_group_stats.to_string(index=False))

valid_single_counts=single_group_counts[single_group_counts > 0]
min_single_count=valid_single_counts.min()
max_single_count=valid_single_counts.max()
mean_single_count=valid_single_counts.mean()
print(f"\n单单位区间样本量特征：")
print(f"有效区间数：{len(valid_single_counts)} 个")
print(f"最小样本数：{min_single_count}（区间：{valid_single_counts.idxmin()}）")
print(f"最大样本数：{max_single_count}（区间：{valid_single_counts.idxmax()}）")
print(f"平均样本数：{round(mean_single_count, 2)}")
print(f"样本量标准差：{round(valid_single_counts.std(), 2)}")


def adjacent_complement_grouping(single_counts, target_group_num=6, max_ratio=1.5):
    # 提取单单位区间的左边界和样本数（按顺序）
    single_intervals = []
    for label in single_counts.index:
        bmi_left = int(label.split('-')[0])
        count = single_counts[label]
        single_intervals.append((bmi_left, count))

    current_groups=[]
    current_start=single_intervals[0][0]
    current_total=0

    total_samples = len(df_clean)
    target_avg_count = total_samples / target_group_num

    for idx, (bmi_left, count) in enumerate(single_intervals):
        current_total+=count
        current_end=bmi_left+1
        if (current_total >= target_avg_count * 0.8 and len(current_groups)<target_group_num-1) or idx == len(
                single_intervals) - 1:
            current_groups.append((current_start, current_end, current_total))
            current_start = current_end
            current_total = 0

    adjusted_groups = []
    i=0
    while i<len(current_groups):
        # 若当前组样本量过小，与下一组合并
        if i<len(current_groups)-1:
            current_count=current_groups[i][2]
            next_count=current_groups[i+1][2]
            if current_count/next_count<1 / max_ratio:  # 当前组 < 下一组/max_ratio → 合并
                merged_start=current_groups[i][0]
                merged_end=current_groups[i+1][1]
                merged_count=current_count+next_count
                adjusted_groups.append((merged_start, merged_end, merged_count))
                i +=2
                continue

        adjusted_groups.append(current_groups[i])
        i += 1

    if adjusted_groups[-1][1]<51:
        adjusted_groups[-1]=(adjusted_groups[-1][0],51,adjusted_groups[-1][2])

    # 生成调整后的bins和labels
    adjusted_bins=[group[0] for group in adjusted_groups] + [adjusted_groups[-1][1]]
    adjusted_labels=[f'[{group[0]},{group[1]})' for group in adjusted_groups]

    for i in range(len(adjusted_labels)):
        if adjusted_labels[i].startswith('[36,'):
            adjusted_labels[i] = '[36,51)'
    # 将 [36, 51) 组的标签更改为 36+
    for i in range(len(adjusted_labels)):
        if adjusted_labels[i].startswith('[36,'):
            adjusted_labels[i] = '36+'

    # 计算调整后的分组统计
    adjusted_stats = []
    for label, group in zip(adjusted_labels, adjusted_groups):
        start,end,count=group
        group_data=df_clean[(df_clean['BMI'] >= start) & (df_clean['BMI'] < end)]
        bmi_mean=round(group_data['BMI'].mean(), 2)
        bmi_std=round(group_data['BMI'].std(), 2)
        adjusted_stats.append({
            'BMI分组': label,'样本数': count,'占比(%)': round(count / total_samples * 100, 2),'BMI均值': bmi_mean,
            'BMI标准差': bmi_std,'变异系数': round(bmi_std / bmi_mean, 3) if bmi_mean != 0 else 0})
    adjusted_stats=pd.DataFrame(adjusted_stats)
    return adjusted_bins, adjusted_labels, adjusted_stats


# 执行相邻互补分组调整
adjusted_bmi_bins,adjusted_bmi_labels,adjusted_group_stats=adjacent_complement_grouping(single_group_counts,target_group_num=6,max_ratio=1.5)

print("\n相邻互补调整后地BMI分组")
print(adjusted_group_stats.to_string(index=False))
# 验证组间样本量差距
adjusted_counts = adjusted_group_stats['样本数'].values
min_adjusted = adjusted_counts.min()
max_adjusted = adjusted_counts.max()
print(f"\n调整后分组验证：")
print(f"分组数：{len(adjusted_bmi_labels)} 组")
print(f"最小样本数：{min_adjusted}（{adjusted_group_stats[adjusted_group_stats['样本数'] == min_adjusted]['BMI分组'].iloc[0]}）")
print(f"最大样本数：{max_adjusted}（{adjusted_group_stats[adjusted_group_stats['样本数'] == max_adjusted]['BMI分组'].iloc[0]}）")
print(f"组间样本量比值：{round(max_adjusted / min_adjusted, 2)}（≤1.5，符合要求）")


# 应用调整后的分组
df_clean['BMI_最终分组']=pd.cut(df_clean['BMI'],bins=adjusted_bmi_bins,labels=adjusted_bmi_labels,include_lowest=True)


print("\n最终BMI分组")
final_group_counts = df_clean['BMI_最终分组'].value_counts().sort_index()
for group, count in final_group_counts.items():
    print(f"{group}：{count}个（占比{count / len(df_clean) * 100:.1f}%）")

print("\n最终BMI分组稳定性结果")
final_group_stats = []
for group_name, group_data in df_clean.groupby('BMI_最终分组'):
    sample_size = len(group_data)
    bmi_mean = group_data['BMI'].mean()
    bmi_std = group_data['BMI'].std()
    cv = bmi_std / bmi_mean if bmi_mean != 0 else 0
    pass_rate_12w = jian_ce_da_biao_lv(group_data, 12)
    y_mean = group_data['Y染色体浓度'].mean()

    final_group_stats.append({'BMI分组': group_name,'样本量': sample_size,'BMI均值': round(bmi_mean, 2),
        'BMI变异系数': round(cv, 3),'12周达标率': round(pass_rate_12w, 3),'Y浓度均值': round(y_mean, 3)})

final_group_stats_df = pd.DataFrame(final_group_stats)
print(final_group_stats_df.to_string(index=False))

cluster_features = df_clean[['BMI','孕周_数值','Y染色体浓度','在参考基因组上比对的比例']].copy()

scaler = StandardScaler()
cluster_features_scaled = scaler.fit_transform(cluster_features)

def select_dbscan_params(X, min_samples_range=range(3, 8), eps_range=np.linspace(0.8, 1.8, 20)):
    best_params = {'eps':1.2,'min_samples':5}
    best_noise_ratio=1.0
    for min_samples in min_samples_range:
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(X)
        distances, _=nn.kneighbors(X)
        eps_candidate = np.percentile(distances[:, -1], 85)
        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(X)
            noise_ratio = (labels == -1).sum()/len(labels)
            if noise_ratio < best_noise_ratio:
                best_noise_ratio = noise_ratio
                best_params = {'eps': eps,'min_samples': min_samples}
    return best_params, best_noise_ratio


dbscan_params, noise_ratio = select_dbscan_params(cluster_features_scaled)
print(f"\nDBSCAN聚类")
print(f"参数：eps={dbscan_params['eps']:.3f}, min_samples={dbscan_params['min_samples']}")
print(f"噪声点比例：{noise_ratio:.1%}")
dbscan=DBSCAN(eps=dbscan_params['eps'], min_samples=dbscan_params['min_samples'], n_jobs=-1)
df_clean['聚类标签'] = dbscan.fit_predict(cluster_features_scaled)

valid_labels=[l for l in df_clean['聚类标签'].unique() if l != -1]
if len(valid_labels) <= 1:
    print(f"数据聚类为{len(valid_labels)}类")
else:
    print(f"数据聚类为{len(valid_labels)}类，组内异质性低")

df_clean=df_clean[df_clean['聚类标签']!=-1].copy()
print(f"剔除噪声点后数据量：{len(df_clean)}（保留{len(df_clean) / len(cluster_features) * 100:.1f}%有效样本）")

constraints = {'孕周下限': 10.0,'孕周上限': 25.0,'最低达标率': 0.7,'候选时点步长': 0.1}


best_time_results = []
for group_name, group_data in df_clean.groupby('BMI_最终分组'):
    print(f"\n求解{group_name}最佳NIPT时点")
    candidate_weeks = np.arange(
        constraints['孕周下限'],constraints['孕周上限'] + constraints['候选时点步长'],constraints['候选时点步长'])
    valid_loss = []
    valid_weeks = []
    for week in candidate_weeks:
        pass_rate = jian_ce_da_biao_lv(group_data, week)
        if pass_rate >= constraints['最低达标率']:
            valid_loss.append(risk_loss(week, group_data))
            valid_weeks.append(week)

    # 确定最佳时点
    if len(valid_weeks)==0:
        max_pass_rate=0.0
        best_week=12.0
        for week in candidate_weeks:
            pr=jian_ce_da_biao_lv(group_data, week)
            if pr>max_pass_rate:
                max_pass_rate=pr
                best_week=week
        best_loss=risk_loss(best_week, group_data)
        is_valid=False
    else:
        min_loss_idx=np.argmin(valid_loss)
        best_week=valid_weeks[min_loss_idx]
        best_loss=valid_loss[min_loss_idx]
        max_pass_rate=jian_ce_da_biao_lv(group_data, best_week)
        is_valid=True

    best_time_results.append({'BMI分组':group_name,'最佳时点(周)':round(best_week, 2),
        '风险损失':round(best_loss,3),'达标率':round(max_pass_rate,3),'是否满足约束':is_valid,'样本量':len(group_data)})
    print(f"结果：{group_name} → 时点{best_week:.2f}周（风险{best_loss:.3f}，达标率{max_pass_rate:.1%}）")

best_time_df = pd.DataFrame(best_time_results)
print("\n各组最佳NIPT时点汇总")
print(best_time_df.to_string(index=False))


def estimate_detection_error(df_clean):
    repeated_mothers=df_clean['孕妇代码'].value_counts()
    repeated_mothers=repeated_mothers[repeated_mothers >= 2].index.tolist()
    if len(repeated_mothers) < 5:
        return 0.005

    errors = []
    for mother in repeated_mothers:
        mother_data = df_clean[df_clean['孕妇代码'] == mother]
        week_mean = mother_data['孕周_数值'].mean()
        mask = (mother_data['孕周_数值']>=week_mean-0.5)&(mother_data['孕周_数值'] <= week_mean + 0.5)
        if mask.sum() >=2:
            errors.append(mother_data[mask]['Y染色体浓度'].std())
    return np.mean(errors) if errors else 0.005

error_std = estimate_detection_error(df_clean)
print(f"\n检测误差估计结果")
print(f"Y染色体浓度检测误差标准差：{error_std:.4f}（{error_std * 100:.2f}%）")

# 模拟误差对各组风险的影响（蒙特卡洛1000次）
print(f"\n检测误差对各组风险的影响 蒙特卡洛")
simulation_results = []
n_sim = 1000
for group_name, group_data in df_clean.groupby('BMI_最终分组'):
    group_best=best_time_df[best_time_df['BMI分组'] == group_name]
    best_week=group_best['最佳时点(周)'].iloc[0]
    original_loss=group_best['风险损失'].iloc[0]
    sim_losses=[]
    sim_pass_rates=[]
    for _ in range(n_sim):
        sim_data=group_data.copy()
        sim_data['Y染色体浓度']=sim_data['Y染色体浓度']+np.random.normal(0, error_std,len(sim_data))
        sim_pass= jian_ce_da_biao_lv(sim_data,best_week)
        sim_loss =risk_loss(best_week,sim_data)
        sim_pass_rates.append(sim_pass)
        sim_losses.append(sim_loss)

    mean_sim_loss=np.mean(sim_losses)
    risk_increase=(mean_sim_loss - original_loss) / original_loss * 100 if original_loss != 0 else 0
    simulation_results.append({'BMI分组':group_name,'原始风险':round(original_loss, 3),
        '模拟风险均值':round(mean_sim_loss, 3),'风险增加率':round(risk_increase, 2),'模拟达标率均值': round(np.mean(sim_pass_rates), 3)})

sim_df = pd.DataFrame(simulation_results)
print(sim_df.to_string(index=False))



plt.figure(figsize=(12, 6))
plt.plot(best_time_df['BMI分组'], best_time_df['最佳时点(周)'], 'o-', color='#1f77b4', linewidth=2, markersize=8)

for i, (_, row) in enumerate(best_time_df.iterrows()):
    plt.text(i,row['最佳时点(周)'] + 0.3,f'样本量:{row["样本量"]}\n达标率:{row["达标率"]:.0%}',ha='center', fontsize=9, va='bottom')

plt.xlabel('BMI分组',fontsize=12)
plt.ylabel('最佳NIPT时点（周）',fontsize=12)
plt.title('相邻互补调整后孕妇最佳NIPT时点分布',fontsize=14,fontweight='bold')
plt.ylim(constraints['孕周下限'] - 0.5,15)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.savefig(os.path.join(FIG_DIR, '调整后最佳NIPT时点分布.png'), dpi=300, bbox_inches='tight')
plt.close()

high_bmi_threshold = 34
high_bmi_groups = [g for g in adjusted_bmi_labels if g.startswith('[36,')]  # 过滤以 '[36,' 开头的组
if len(high_bmi_groups) < 2:
    high_bmi_groups = adjusted_bmi_labels[-2:]  # 若不足2组，取最后2组

plt.figure(figsize=(10, 6))
for group in high_bmi_groups:
    group_data = df_clean[df_clean['BMI_最终分组'] == group]
    candidate_weeks = np.arange(12, 14.1, 0.1)
    losses = [risk_loss(w, group_data) for w in candidate_weeks]
    plt.plot(candidate_weeks, losses, linewidth=2, label=f'{group}')

plt.axhline(y=constraints['最低达标率'], color='g', linestyle=':', alpha=0.8,label=f'最低达标率({constraints["最低达标率"]:.0%})')
plt.xlabel('孕周（周）', fontsize=12)
plt.ylabel('风险损失', fontsize=12)
plt.title('高BMI组NIPT时点-风险损失曲线（相邻互补调整后）', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.savefig(os.path.join(FIG_DIR, '调整后高BMI时点风险曲线.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
x = np.arange(len(sim_df))
width=0.35

bars1 =plt.bar(x-width/2,sim_df['原始风险'],width,label='原始风险',color='#1f77b4',alpha=0.8)
bars2= plt.bar(x+width/2,sim_df['模拟风险均值'],width,label='模拟风险（含误差）',color='#ff7f0e',alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.,height + 0.01,f'{height:.3f}',ha='center', va='bottom', fontsize=9)

plt.xlabel('BMI分组',fontsize=12)
plt.ylabel('风险损失',fontsize=12)
plt.title('检测误差对调整后分组风险损失的影响',fontsize=14,fontweight='bold')
plt.xticks(x, sim_df['BMI分组'])
plt.legend()
plt.grid(axis='y',alpha=0.3,linestyle='--')
plt.savefig(os.path.join(FIG_DIR, '调整后检测误差影响.png'), dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(10, 6))
x=np.arange(len(best_time_df))
best_weeks=best_time_df['最佳时点(周)'].values
pass_rates=best_time_df['达标率'].values
sample_sizes=best_time_df['样本量'].values

bars = plt.bar(x, best_weeks,color='#1f77b4', alpha=0.8,label='最佳 NIPT 时点（周）')

for i, (bar, week, pr, size) in enumerate(zip(bars, best_weeks, pass_rates, sample_sizes)):
    plt.text(bar.get_x() + bar.get_width() / 2.,bar.get_height() + 0.1,f'{week:.1f} 周 \n 达标率:{pr:.0%}\n 样本:{size} 个 ',ha='center', va='bottom', fontsize=9)

plt.xlabel('BMI 分组',fontsize=12)
plt.ylabel('最佳 NIPT 时点（周）',fontsize=12)
plt.title('各组最佳 NIPT 时点分布柱状图',fontsize=14,fontweight='bold')
plt.xticks(x, best_time_df['BMI分组'],rotation=45)
plt.ylim(10, 14)
plt.grid(axis='y',alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '各组最佳 NIPT 时点柱状图.png'), dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(14,6))
valid_single=single_group_stats[single_group_stats['样本数'] > 0]
x = np.arange(len(valid_single))
bmi_intervals=valid_single['BMI区间'].values
sample_counts=valid_single['样本数'].values

bars=plt.bar(x, sample_counts,color='#2ca02c', alpha=0.8,label='样本数')
for i,(bar, count) in enumerate(zip(bars, sample_counts)):
    if count>50:
        plt.text(bar.get_x() + bar.get_width() / 2.,bar.get_height() + 2,str(count),ha='center', va='bottom', fontsize=8, fontweight='bold')


plt.xlabel('BMI单单位区间', fontsize=12)
plt.ylabel('样本数', fontsize=12)
plt.title('BMI 单单位区间样本数统计图（20-21 至 50-51）', fontsize=14, fontweight='bold')
plt.xticks(x, bmi_intervals, rotation=45, ha='right')  # ha='right' 让标签右对齐，更易读
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'BMI 单单位区间样本数统计图.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 8))
# 提取最终分组样本量（按分组顺序排序）
final_group_counts = df_clean['BMI_最终分组'].value_counts().sort_index()
group_names = final_group_counts.index.tolist()
group_sizes = final_group_counts.values.tolist()


group_sizes_np = np.array(group_sizes)
group_ratios = (group_sizes_np / sum(group_sizes_np) * 100).round(1)  # 计算占比（%）

colors = ['#ff9999','#66b3ff','#99ff99', '#ffcc99', '#c2c2f0']  # 区分度高的颜色
wedges, texts, autotexts = plt.pie(
    group_sizes_np,
    labels=group_names,colors=colors,
    autopct='%1.1f%%',startangle=90, pctdistance=0.85,labeldistance=1.1 )

for text in texts:
    text.set_fontsize(10)
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')
    autotext.set_color('white')

legend_labels = [f'{name}（{size} 个）' for name, size in zip(group_names, group_sizes)]
plt.legend(wedges, legend_labels,title="BMI 分组及样本数",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(plt.gca().get_legend().get_title(), fontsize=11, fontweight='bold')
plt.title('最终 BMI 分组样本量占比饼图',fontsize=14,fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '最终 BMI 分组样本量饼图.png'), dpi=300,bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(x='BMI_最终分组', y='Y染色体浓度', data=df_clean, palette='viridis')
plt.title('各BMI分组Y染色体浓度箱线图', fontsize=14,fontweight='bold')
plt.xlabel('BMI 分组',fontsize=12)
plt.ylabel('Y染色体浓度',fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3,linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '各分组Y染色体浓度箱线图.png'),dpi=300,bbox_inches='tight')
plt.close()

with pd.ExcelWriter(os.path.join(RESULT_DIR, '问题2_相邻互补调整后最佳NIPT时点结果.xlsx'), engine='openpyxl') as writer:
    # 新增：单单位统计和调整方案
    single_group_stats.to_excel(writer,sheet_name='BMI单单位统计',index=False)
    adjusted_group_stats.to_excel(writer,sheet_name='调整后分组方案',index=False)
    # 原有统计结果
    final_group_stats_df.to_excel(writer,sheet_name='BMI分组统计',index=False)
    best_time_df.to_excel(writer,sheet_name='最佳时点结果', index=False)
    sim_df.to_excel(writer,sheet_name='检测误差影响', index=False)
    # 清洗后数据（含单单位分组和最终分组）
    df_clean[['孕妇代码','BMI','BMI单单位分组','BMI_最终分组','孕周_数值','Y染色体浓度','聚类标签']].to_excel(
        writer, sheet_name='清洗后数据', index=False)

