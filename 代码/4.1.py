import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import (precision_recall_curve,average_precision_score,precision_score,recall_score,
            f1_score, roc_auc_score,confusion_matrix, accuracy_score,classification_report,roc_curve)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from typing import Union, List, Dict, Tuple
import xgboost as xgb

plt.switch_backend('Agg')
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
# --- 路径配置：按项目目录结构管理输入输出 ---
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE, '..', '数据与题目', '附件.xlsx')
FIG_DIR = os.path.join(_BASE, '..', '图片', '问题4')
RESULT_DIR = os.path.join(_BASE, '..', '结果文件')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams['mathtext.fontset'] = 'dejavusans'

def yunzhou(week_str:Union[str,float,int]) -> float:
    if pd.isna(week_str):return np.nan
    week_str=str(week_str).strip()
    if 'w' in week_str:
        try:
            w_part, rest = week_str.split('w', 1)
            weeks = int(w_part)
            if '+' in rest:
                d_part = rest.split('+')[1].strip()
                weeks += int(d_part) / 7
            return round(weeks, 3)
        except (ValueError, IndexError):
            warnings.warn(f"无法解析孕周格式：{week_str}，返回NaN")
            return np.nan
    try:
        return round(float(week_str), 3)
    except ValueError:
        warnings.warn(f"无法解析孕周格式：{week_str}，返回NaN")
        return np.nan


def huaiyuncishu(count:Union[str, int]) -> int:
    if pd.isna(count):return np.nan
    count_str=str(count).strip()
    return 3 if count_str == '≥3' else int(count_str)

def p_ivf(ivf_str: str) -> int:
    if pd.isna(ivf_str):return np.nan
    ivf_str=str(ivf_str).strip()
    return 1 if any(keyword in ivf_str for keyword in ['IVF','试管婴儿','IUI']) else 0

def process_target(abnormal_str: str) -> int:
    if pd.isna(abnormal_str) or str(abnormal_str).strip() == '':return 0
    abnormal_types=str(abnormal_str).strip().split('T')
    for typ in abnormal_types:
        if typ in ['13','18','21']:return 1
    return 0


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建测序质量派生特征：有效读段、Z值可靠性、GC偏离度等"""
    df = df.copy()
    df['有效读段比例']=1-df['被过滤掉读段数的比例']
    df['有效读段数_估计']=df['原始读段数']*df['有效读段比例']
    df['原始读段数_对数']=np.log1p(df['原始读段数'])
    df['唯一比对的读段数_比例']=df['唯一比对的读段数']/df['原始读段数']
    df['唯一比对_有效比例']= df['唯一比对的读段数']/df['有效读段数_估计']

    for chrom in ['13','18','21','X']:
        df[f'{chrom}号Z值_绝对值']=np.abs(df[f'{chrom}号染色体的Z值'])
    for chrom in ['13','18','21']:
        df[f'{chrom}号Z值_可靠度']= df[f'{chrom}号Z值_绝对值'] *df['有效读段比例']

    normal_gc_range=(0.4,0.6)
    df['全局GC_偏离度']=np.abs(df['GC含量']-np.mean(normal_gc_range))
    df['全局GC_异常标志']=((df['GC含量']<normal_gc_range[0])|(df['GC含量'] >normal_gc_range[1])).astype(int)
    for chrom in ['13','18','21']:
        df[f'{chrom}号GC_偏离度'] =np.abs(df[f'{chrom}号染色体的GC含量']- np.mean(normal_gc_range))

    df['X染色体浓度_标准化']= (df['X染色体浓度']-df['X染色体浓度'].mean())/df['X染色体浓度'].std()
    df['XZ_与_GC交互']= df['X号Z值_绝对值']*df['全局GC_偏离度']
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df['怀孕次数_分组']=pd.cut(df['怀孕次数_数值'],bins=[0,1,2,np.inf],labels=['1次', '2次', '≥3次'], right=False)
    ohe=OneHotEncoder(sparse_output=False, drop='first',dtype=int)
    ohe_features=ohe.fit_transform(df[['怀孕次数_分组']])
    ohe_df =pd.DataFrame(ohe_features,columns=[f'怀孕次数_{label}' for label in ohe.categories_[0][1:]],index=df.index)
    df=pd.concat([df, ohe_df], axis=1)
    return df

def data(file_path: str, sheet_name: str = '女胎检测数据') -> pd.DataFrame:
    """完整预处理流程：数据加载→格式转换→特征工程→缺失值填充"""
    core_cols =['序号', '孕妇代码', '年龄', '身高', '体重', '孕妇BMI', '检测孕周',
        '怀孕次数', '生产次数', 'IVF妊娠', '原始读段数', '在参考基因组上比对的比例','重复读段的比例', '唯一比对的读段数', 'GC含量', '被过滤掉读段数的比例',
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X号染色体的Z值', 'X染色体浓度','13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', '染色体的非整倍体']
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=core_cols, engine='openpyxl')
    print(f"原始数据：{df.shape} 行×列，孕妇数量：{df['孕妇代码'].nunique()}")
    df['检测孕周_数值']=df['检测孕周'].apply(yunzhou)
    df['怀孕次数_数值']=df['怀孕次数'].apply(huaiyuncishu)
    df['生产次数_数值']= df['生产次数'].apply(lambda x: int(x) if pd.notna(x) else np.nan)
    df['IVF_编码'] =df['IVF妊娠'].apply(p_ivf)
    df['目标变量_异常']= df['染色体的非整倍体'].apply(process_target)
    df = create_features(df)
    df = encode_features(df)

    numeric_cols=[col for col in df.columns if df[col].dtype in [np.float64, np.int64] and col not in ['目标变量_异常']]
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby('孕妇代码')[col].transform(lambda x: x.fillna(x.mean()))
            df[col].fillna(df[col].median(), inplace=True)
    print(f"\n预处理后目标变量分布：正常={df['目标变量_异常'].value_counts()[0]}，异常={df['目标变量_异常'].value_counts()[1]}")
    return df

def fenzu(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42) -> Tuple[
    pd.DataFrame,pd.DataFrame]:   #按孕妇分组划分训练/测试集，确保同孕妇样本在同一集合
    # 兼容性修复(pandas>=3.0)：unique() 返回 Arrow 数组，需转 numpy 供 sklearn 索引
    unique_patients=np.asarray(df['孕妇代码'].unique())
    # 兼容性修复(pandas>=3.0)：stratify 传入 numpy 数组而非 Series，并按 unique_patients 顺序对齐
    patient_labels=df.groupby('孕妇代码')['目标变量_异常'].max()
    stratify_labels=patient_labels.reindex(unique_patients).to_numpy()
    train_patients,test_patients=train_test_split(unique_patients, test_size=test_size, random_state=random_state,
        stratify=stratify_labels)
    df_train=df[df['孕妇代码'].isin(train_patients)].copy()
    df_test=df[df['孕妇代码'].isin(test_patients)].copy()
    print(f"训练集：{df_train.shape[0]}样本，{df_train['孕妇代码'].nunique()}孕妇，异常比例{df_train['目标变量_异常'].mean():.1%}")
    print(f"测试集：{df_test.shape[0]}样本，{df_test['孕妇代码'].nunique()}孕妇，异常比例{df_test['目标变量_异常'].mean():.1%}")
    return df_train, df_test

def calculate_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
    """双阈值优化：高召回率阈值（≥95%）和F2最优阈值"""
    precision,recall,thresholds=precision_recall_curve(y_true, y_proba)
    precision,recall=precision[:-1], recall[:-1]
    beta=2
    f2_scores=(1+beta**2)*(precision*recall)/(beta**2*precision+recall+1e-7)
    high_recall_indices=np.where(recall>=0.95)[0]
    high_recall_threshold=thresholds[np.argmax(precision[high_recall_indices])] if high_recall_indices.size > 0 else \
    thresholds[np.argmax(f2_scores)]
    f2_opt_threshold = thresholds[np.argmax(f2_scores)]

    return {'高召回率阈值':high_recall_threshold,'F2最优阈值':f2_opt_threshold},pd.DataFrame({'阈值': thresholds, '精确率': precision, '召回率': recall, 'F2分数': f2_scores}).round(4)


def train_model(df_train: pd.DataFrame, model_name: str) -> Tuple[Pipeline, List[str], StandardScaler]:
    feature_cols = ['年龄','身高','体重','孕妇BMI','检测孕周_数值','生产次数_数值','IVF_编码','原始读段数_对数', '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数_比例',
        '有效读段比例', '全局GC_偏离度', '全局GC_异常标志','13号Z值_可靠度', '18号Z值_可靠度', '21号Z值_可靠度', 'X号Z值_绝对值',
        '13号GC_偏离度', '18号GC_偏离度','21号GC_偏离度','X染色体浓度_标准化', 'XZ_与_GC交互','怀孕次数_2次', '怀孕次数_≥3次']
    feature_cols=[col for col in feature_cols if col in df_train.columns]
    X=df_train[feature_cols]
    y=df_train['目标变量_异常']
    groups=df_train['孕妇代码']
    print(f"\n开始训练 {model_name} 模型：使用{len(feature_cols)}个特征，样本数{X.shape[0]}")
    scaler = StandardScaler()
    #pipeline = None
    #param_grid = None

    if model_name =='RandomForest':
        model= RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        param_grid= {'model__n_estimators': [100, 200],'model__max_depth': [5, 8, 12],'model__min_samples_leaf': [1, 2, 4]}
        pipeline =Pipeline([('scaler', scaler), ('model', model)])
    elif model_name=='XGBoost':
        model=xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
        param_grid={'model__n_estimators': [100, 200],'model__max_depth': [3, 5, 7],'model__learning_rate': [0.05, 0.1]}
        pipeline=Pipeline([('scaler', scaler), ('model', model)])

    elif model_name=='GradientBoosting':
        model=GradientBoostingClassifier(random_state=42,min_samples_leaf=5,subsample=0.8,validation_fraction=0.1,n_iter_no_change=5)
        param_grid= {'model__n_estimators': [100, 200],'model__max_depth': [3, 5, 7],'model__learning_rate': [0.05, 0.1, 0.2]}
        pipeline =Pipeline([('scaler', scaler), ('model', model)])
    else:
        raise ValueError("支持的模型：'RandomForest', 'XGBoost', 'GradientBoosting'")
    if pipeline is None or param_grid is None:
        raise ValueError(f"模型 {model_name} 初始化失败")
    group_cv = GroupKFold(n_splits=5)
    # 兼容性修复：cv 生成器物化为列表、n_jobs=1 避免 joblib pickle 失败（数据量小，影响可忽略）
    grid_search = GridSearchCV(
        estimator=pipeline,param_grid=param_grid,cv=list(group_cv.split(X, y, groups)),scoring='roc_auc',n_jobs=1,verbose=1)
    grid_search.fit(X, y)
    best_pipeline = grid_search.best_estimator_
    print(f"\n{model_name} 最优参数：{grid_search.best_params_}")
    print(f"{model_name} 交叉验证平均召回率：{grid_search.best_score_:.3f}")

    if model_name in ['RandomForest', 'XGBoost', 'GradientBoosting']:
        final_model = best_pipeline.named_steps['model']
        if hasattr(final_model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({'特征名称': feature_cols,'重要性': final_model.feature_importances_}).sort_values('重要性', ascending=False).round(4)
            print(f"\n{model_name} - Top10关键特征：")
            print(feature_importance_df.head(10))
            plt.figure(figsize=(10, 6))
            sns.barplot(x='重要性',y='特征名称',data=feature_importance_df.head(10), palette='viridis')
            plt.title(f'{model_name}-Top10 特征重要性', fontsize=12)
            plt.savefig(os.path.join(FIG_DIR, f'{model_name}_特征重要性.png'), dpi=300)
            plt.close()
    return best_pipeline, feature_cols, scaler

def traditional_z_rule(df: pd.DataFrame) -> np.ndarray:
    z_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值']
    return ((np.abs(df[z_cols])>3).any(axis=1)).astype(int)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str) -> Dict:
    beta=2
    precision=precision_score(y_true, y_pred, zero_division=0)
    recall=recall_score(y_true, y_pred, zero_division=0)
    f2 = (1+beta**2)*(precision * recall) / (beta ** 2 * precision + recall + 1e-7)
    return {
        '模型名称': model_name,
        '准确率': round(accuracy_score(y_true, y_pred), 3),
        '精确率': round(precision,3),
        '召回率': round(recall,3),
        'F1分数': round(f1_score(y_true, y_pred, zero_division=0), 3),
        'F2分数': round(f2, 3),
        'ROC-AUC': round(roc_auc_score(y_true, y_proba), 3) if y_proba is not None else np.nan,
        'PR-AUC': round(average_precision_score(y_true, y_proba), 3) if y_proba is not None else np.nan}


def compare_models(df_test: pd.DataFrame, models: Dict[str, Pipeline], feature_cols: List[str]) -> pd.DataFrame:
    """对比多个模型与传统Z值规则，保存P-R曲线和混淆矩阵"""
    X_test = df_test[feature_cols]
    y_test = df_test['目标变量_异常']
    all_metrics = []
    plot_data = {}
    colors = {'RandomForest': '#2E86AB', 'XGBoost': '#A23B72', 'GradientBoosting': '#F18F01', 'Z值规则': '#3E8914'}

    for name, pipeline in models.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        thresholds, _ = calculate_thresholds(y_test, y_proba)
        y_pred_f2 = (y_proba >= thresholds['F2最优阈值']).astype(int)
        metrics = evaluate_model(y_test, y_pred_f2, y_proba, f'{name} (F2阈值={thresholds["F2最优阈值"]:.3f})')
        all_metrics.append(metrics)

        metrics_clean_name = evaluate_model(y_test, y_pred_f2, y_proba, name)
        plot_data[name] = {'proba': y_proba, 'metrics': metrics_clean_name}

    y_pred_z = traditional_z_rule(df_test)
    metrics_z = evaluate_model(y_test, y_pred_z, y_pred_z, '传统Z值规则 (|Z|>3)')
    all_metrics.append(metrics_z)

    metrics_z_clean = evaluate_model(y_test, y_pred_z, y_pred_z, 'Z值规则')
    plot_data['Z值规则'] = {'proba': y_pred_z, 'metrics': metrics_z_clean}

    compare_df = pd.DataFrame(all_metrics).sort_values('F2分数',ascending=False)
    print("\n模型性能对比")
    print(compare_df.to_string(index=False))


    metrics_for_plot = pd.DataFrame([data['metrics'] for data in plot_data.values()])
    metrics_melted = metrics_for_plot.melt(id_vars='模型名称',value_vars=['准确率','精确率','召回率','F2分数'],var_name='指标', value_name='分数')

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=metrics_melted, x='模型名称', y='分数', hue='指标', palette='muted')
    plt.title('各模型核心性能指标对比', fontsize=16)
    plt.ylabel('指标得分', fontsize=12)
    plt.xlabel('模型', fontsize=12)
    plt.xticks(rotation=10, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(title='性能指标', fontsize=10)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',(p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='center',fontsize=9, color='black',
                    xytext=(0, 5),textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '模型性能对比柱状图.png'), dpi=300)
    plt.close()

    # P-R曲线和ROC曲线合并绘制
    fig,(ax1, ax2)=plt.subplots(1,2,figsize=(16, 7))
    for name, data in plot_data.items():
        precision, recall, _ = precision_recall_curve(y_test, data['proba'])
        ax1.plot(recall, precision, lw=2, label=f'{name} (PR-AUC={data["metrics"]["PR-AUC"]:.3f})', color=colors[name],
                 linestyle='--' if name == 'Z值规则'else '-')
        fpr, tpr, _ = roc_curve(y_test, data['proba'])
        ax2.plot(fpr, tpr, lw=2, label=f'{name} (ROC-AUC={data["metrics"]["ROC-AUC"]:.3f})', color=colors[name],
                 linestyle='--' if name == 'Z值规则' else '-')
    ax1.set_title('P-R 曲线对比', fontsize=14)
    ax1.set_xlabel('召回率 (Recall) - 减少漏诊')
    ax1.set_ylabel('精确率 (Precision) - 减少误诊')
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    ax2.set_title('ROC 曲线对比', fontsize=14)
    ax2.set_xlabel('假阳性率 (FPR)')
    ax2.set_ylabel('真阳性率 (TPR)')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'PR_and_ROC_curves.png'), dpi=300)
    plt.close()

    best_model_name = compare_df.iloc[0]['模型名称'].split(' ')[0]
    if best_model_name in models:
        best_pipeline = models[best_model_name]
        y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
        best_thresholds, _ = calculate_thresholds(y_test, y_proba_best)
        y_pred_best = (y_proba_best >= best_thresholds['F2最优阈值']).astype(int)

        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
        plt.title(f'最优模型({best_model_name})混淆矩阵', fontsize=12)
        plt.xlabel('预测标签', fontsize=11)
        plt.ylabel('实际标签', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'最优模型_{best_model_name}_混淆矩阵.png'), dpi=300)
        plt.close()
    return compare_df


def predict_single_sample(input_df: pd.DataFrame, best_pipeline: Pipeline, feature_cols: List[str],
                          thresholds: Dict) -> Dict:
    ohe = OneHotEncoder(sparse_output=False, drop='first', dtype=int)
    if '怀孕次数_分组' not in input_df.columns:
        input_df['怀孕次数_分组']=pd.cut(input_df['怀孕次数_数值'], bins=[0, 1, 2, np.inf],labels=['1次','2次','≥3次'], right=False)
        ohe_features=ohe.fit_transform(input_df[['怀孕次数_分组']])
        ohe_df=pd.DataFrame(ohe_features,columns=[f'怀孕次数_{label}' for label in ['2次', '≥3次']],index=input_df.index)
        input_df=pd.concat([input_df, ohe_df],axis=1)

    X_input=input_df.reindex(columns=feature_cols)
    y_proba=best_pipeline.predict_proba(X_input)[0,1]
    risk_level,suggestion=get_risk_strategy(y_proba, thresholds['高召回率阈值'])

    return {'样本信息':{'孕妇代码':str(input_df['孕妇代码'].iloc[0])},
        '判定结果':{'异常概率':round(float(y_proba),3),'风险等级': risk_level,
        '判定(F2最优)':'异常' if y_proba >= thresholds['F2最优阈值'] else '正常'},'临床建议': suggestion}


def get_risk_strategy(prob: float,threshold_high:float,threshold_low:float=0.3) -> Tuple[str, str]:
    if prob>=threshold_high:
        return '高风险','建议立即安排羊水穿刺等确诊检测。'
    elif threshold_low <= prob < threshold_high:
        return '中风险','建议1-2周后复查NIPT，并密切监测。'
    else:
        return '低风险', '按常规产检流程即可。'


def main(file_path: str) -> None:
    df=data(file_path)
    df_train, df_test =fenzu(df)
    models_to_train = ['RandomForest','XGBoost','GradientBoosting']
    trained_models = {}
    feature_cols = []
    for model_name in models_to_train:
        pipeline, f_cols, _ = train_model(df_train, model_name)
        trained_models[model_name] = pipeline
        if not feature_cols:
            feature_cols = f_cols
    compare_df=compare_models(df_test, trained_models, feature_cols)

    best_model_row=compare_df.iloc[0]
    best_model_name=best_model_row['模型名称'].split(' ')[0]
    best_pipeline=trained_models[best_model_name]
    print(f"最优模型为:{best_model_name} (测试集F2分数:{best_model_row['F2分数']:.3f})")

    y_proba_best = best_pipeline.predict_proba(df_test[feature_cols])[:,1]
    best_thresholds, _ =calculate_thresholds(df_test['目标变量_异常'], y_proba_best)
    sample_data=df_test.iloc[[0]].copy()
    sample_result=predict_single_sample(sample_data, best_pipeline, feature_cols, best_thresholds)
    for key, val in sample_result.items():
        print(f"{key}：{val}")





if __name__=="__main__":
    main(file_path=DATA_PATH)
