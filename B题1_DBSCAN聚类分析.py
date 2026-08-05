# -*- coding: utf-8 -*-
"""
B题 问题1：MIMAC分子数据 DBSCAN + 活性加权聚类分析
数学建模 B 题 问题1：DBSCAN 聚类与活性加权区域识别
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os, sys

# ========== 路径配置 ==========
DATA_DIR = os.environ.get(
    "B1_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
CHARTS_DIR = os.environ.get(
    "B1_CHARTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts"))
os.makedirs(CHARTS_DIR, exist_ok=True)

# ========== 1. 数据读取 ==========
print("=" * 60)
print("阶段1：数据读取与探索性分析")
print("=" * 60)

# 尝试多种编码
for enc in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
    try:
        # skiprows=[1] 跳过中文列名行（第2行），保留英文列名（第1行）
        df = pd.read_csv(os.path.join(DATA_DIR, "附件1：molecular_interaction_manifest.csv"),
                         skiprows=[1], encoding=enc)
        edges = pd.read_csv(os.path.join(DATA_DIR, "附件2：knn_graph_edges.csv"),
                           skiprows=[1], encoding=enc)
        print(f"成功使用编码: {enc}")
        break
    except (UnicodeDecodeError, Exception):
        continue

# 确认列名
print(f"附件1列名: {list(df.columns)}")
print(f"附件2列名: {list(edges.columns)}")

print(f"分子总数: {len(df)}")
print(f"边总数: {len(edges)}")
print(f"\n列名: {list(df.columns)}")
print(f"\n数据描述统计:")
print(df.describe())

# 检查缺失值
print(f"\n缺失值统计:")
print(df.isnull().sum())

# ========== 2. 探索性可视化 ==========
print("\n生成探索性图表...")

# 图1：分子空间分布 + 活性着色
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图：空间散点图（按活性着色）
sc1 = axes[0].scatter(df['Manifold_X'], df['Manifold_Y'],
                      c=df['Bioactivity_Score'], cmap='RdYlGn',
                      s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
axes[0].set_xlabel('Manifold_X', fontsize=11)
axes[0].set_ylabel('Manifold_Y', fontsize=11)
axes[0].set_title('分子二维流形空间分布（按活性着色）', fontsize=12)
plt.colorbar(sc1, ax=axes[0], label='Bioactivity_Score')

# 右图：活性得分直方图
axes[1].hist(df['Bioactivity_Score'], bins=30, color='steelblue',
             edgecolor='white', alpha=0.8)
axes[1].axvline(df['Bioactivity_Score'].median(), color='red',
                linestyle='--', linewidth=1.5, label=f"中位数={df['Bioactivity_Score'].median():.2f}")
axes[1].axvline(df['Bioactivity_Score'].mean(), color='darkorange',
                linestyle='-', linewidth=1.5, label=f"均值={df['Bioactivity_Score'].mean():.2f}")
axes[1].set_xlabel('Bioactivity_Score', fontsize=11)
axes[1].set_ylabel('频数', fontsize=11)
axes[1].set_title('分子活性得分分布', fontsize=12)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'fig1_空间分布与活性分布.png'), dpi=150)
plt.close()
print("  fig1 已保存")

# ========== 3. DBSCAN参数选择：k-距离图 ==========
print("\n" + "=" * 60)
print("阶段2：DBSCAN空间聚类")
print("=" * 60)

coords = df[['Manifold_X', 'Manifold_Y']].values

# k-距离图（k=4, 即min_samples-1）
min_samples = 5
k = min_samples - 1
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(coords)
distances, indices = nbrs.kneighbors(coords)
k_dist = np.sort(distances[:, k])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(k_dist)), k_dist, 'b-', linewidth=1)
ax.set_xlabel('数据点（按距离排序）', fontsize=11)
ax.set_ylabel(f'第{k}近邻距离', fontsize=11)
ax.set_title(f'k-距离图（k={k}，用于确定DBSCAN eps参数）', fontsize=12)
ax.grid(True, alpha=0.3)

# 标注拐点区域
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='eps=1.0')
ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='eps=1.5')
ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='eps=2.0')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'fig2_k距离图.png'), dpi=150)
plt.close()
print("  fig2 已保存")

# 尝试多个eps值，选择最优
print("\n尝试不同eps参数的聚类效果:")
results = []
for eps in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    results.append({'eps': eps, 'n_clusters': n_clusters,
                    'n_noise': n_noise, 'noise_pct': n_noise/len(labels)*100})
    print(f"  eps={eps:.1f}: {n_clusters}个簇, {n_noise}个噪声点({n_noise/len(labels)*100:.1f}%)")

# 选择 eps=1.5（簇数适中，噪声点不过多）
best_eps = 1.5
print(f"\n选定参数: eps={best_eps}, min_samples={min_samples}")

# 最终聚类
dbscan = DBSCAN(eps=best_eps, min_samples=min_samples).fit(coords)
df['cluster'] = dbscan.labels_

n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
n_noise = np.sum(dbscan.labels_ == -1)
print(f"聚类结果: {n_clusters}个有效簇, {n_noise}个噪声点")

# ========== 4. 活性加权：区分"高活性密集区"和"普通区" ==========
print("\n" + "=" * 60)
print("阶段3：活性加权区域识别")
print("=" * 60)

# 计算每个簇的统计指标
cluster_stats = []
for c in sorted(set(dbscan.labels_)):
    if c == -1:
        continue
    mask = df['cluster'] == c
    cluster_stats.append({
        'cluster': c,
        'size': mask.sum(),
        'mean_activity': df.loc[mask, 'Bioactivity_Score'].mean(),
        'median_activity': df.loc[mask, 'Bioactivity_Score'].median(),
        'std_activity': df.loc[mask, 'Bioactivity_Score'].std(),
        'density': mask.sum() / len(df),  # 簇规模占比
        'mean_LogP': df.loc[mask, 'LogP'].mean(),
        'mean_TPSA': df.loc[mask, 'TPSA'].mean(),
        'mean_MolWt': df.loc[mask, 'MolWt'].mean(),
    })

cluster_df_stats = pd.DataFrame(cluster_stats).sort_values('mean_activity', ascending=False)
print("\n各簇活性统计（按平均活性降序）:")
print(cluster_df_stats.to_string(index=False))

# 定义"高活性密集区域"：平均活性 > 整体均值 + 0.5*std 且 簇规模 >= 10
global_mean_act = df['Bioactivity_Score'].mean()
global_std_act = df['Bioactivity_Score'].std()
threshold = global_mean_act + 0.3 * global_std_act  # 略高于均值

print(f"\n整体活性均值={global_mean_act:.3f}, 标准差={global_std_act:.3f}")
print(f"高活性阈值 = 均值+0.3σ = {threshold:.3f}")

high_active_clusters = cluster_df_stats[
    (cluster_df_stats['mean_activity'] >= threshold) &
    (cluster_df_stats['size'] >= 5)
]['cluster'].tolist()

print(f"\n高活性密集区域簇: {high_active_clusters}")
print(f"普通区域簇: {[c for c in cluster_df_stats['cluster'] if c not in high_active_clusters]}")

# 创建区域标签
df['region'] = '普通区域'
df.loc[df['cluster'].isin(high_active_clusters), 'region'] = '高活性密集区域'
df.loc[df['cluster'] == -1, 'region'] = '噪声/离散点'

print(f"\n区域分布:")
print(df['region'].value_counts())

# ========== 5. 聚类结果可视化 ==========
print("\n生成聚类结果图表...")

# 图3：DBSCAN聚类结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
unique_labels = sorted(set(dbscan.labels_))

# 左图：按簇着色
for i, lbl in enumerate(unique_labels):
    mask = df['cluster'] == lbl
    if lbl == -1:
        axes[0].scatter(df.loc[mask, 'Manifold_X'], df.loc[mask, 'Manifold_Y'],
                       c='gray', s=15, alpha=0.4, marker='x', label='噪声')
    else:
        axes[0].scatter(df.loc[mask, 'Manifold_X'], df.loc[mask, 'Manifold_Y'],
                       c=colors_list[i % len(colors_list)], s=30, alpha=0.7,
                       edgecolors='k', linewidth=0.3, label=f'簇{lbl}')
axes[0].set_xlabel('Manifold_X', fontsize=11)
axes[0].set_ylabel('Manifold_Y', fontsize=11)
axes[0].set_title(f'DBSCAN 空间聚类结果 (eps={best_eps}, min_samples={min_samples})', fontsize=12)
axes[0].legend(fontsize=7, loc='upper left', ncol=3)

# 右图：按区域类型着色（高活性密集vs普通）
region_colors = {'高活性密集区域': '#e74c3c', '普通区域': '#3498db', '噪声/离散点': '#95a5a6'}
for reg, color in region_colors.items():
    mask = df['region'] == reg
    axes[1].scatter(df.loc[mask, 'Manifold_X'], df.loc[mask, 'Manifold_Y'],
                   c=color, s=30, alpha=0.7, edgecolors='k',
                   linewidth=0.3, label=reg)
axes[1].set_xlabel('Manifold_X', fontsize=11)
axes[1].set_ylabel('Manifold_Y', fontsize=11)
axes[1].set_title('活性加权区域分类结果', fontsize=12)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'fig3_聚类结果.png'), dpi=150)
plt.close()
print("  fig3 已保存")

# ========== 6. 区域对比分析 ==========
print("\n" + "=" * 60)
print("阶段4：区域对比分析")
print("=" * 60)

high_mask = df['region'] == '高活性密集区域'
normal_mask = df['region'] == '普通区域'

high_df = df[high_mask]
normal_df = df[normal_mask]

print(f"\n高活性密集区域: {len(high_df)}个分子")
print(f"普通区域: {len(normal_df)}个分子")

# 6a. 活性得分对比
print("\n--- 6a. 活性得分对比 ---")
print(f"高活性区 活性均值={high_df['Bioactivity_Score'].mean():.3f} ± {high_df['Bioactivity_Score'].std():.3f}")
print(f"普通区   活性均值={normal_df['Bioactivity_Score'].mean():.3f} ± {normal_df['Bioactivity_Score'].std():.3f}")
t_stat, p_val = stats.ttest_ind(high_df['Bioactivity_Score'], normal_df['Bioactivity_Score'])
print(f"t检验: t={t_stat:.3f}, p={p_val:.6f}")

# 6b. 相似关系对比（用Tanimoto边密度和平均相似度）
print("\n--- 6b. 相似关系对比 ---")
# 构建每个区域的边统计
def calc_edge_stats(mol_ids, edges_df):
    """计算给定分子集合在图中的边统计"""
    src_in = edges_df['Source'].isin(mol_ids)
    tgt_in = edges_df['Target'].isin(mol_ids)
    internal = edges_df[src_in & tgt_in]  # 区域内边
    external = edges_df[src_in | tgt_in]  # 涉及该区域的所有边
    n_mols = len(mol_ids)
    max_possible = n_mols * (n_mols - 1) / 2
    return {
        'n_internal_edges': len(internal),
        'edge_density': len(internal) / max_possible if max_possible > 0 else 0,
        'mean_sim_internal': internal['Tanimoto_Similarity'].mean() if len(internal) > 0 else 0,
        'mean_sim_all': external['Tanimoto_Similarity'].mean() if len(external) > 0 else 0,
    }

high_ids = set(high_df['ID'].values)
normal_ids = set(normal_df['ID'].values)

high_edge = calc_edge_stats(high_ids, edges)
normal_edge = calc_edge_stats(normal_ids, edges)

print(f"高活性区: 内部边{high_edge['n_internal_edges']}条, "
      f"边密度{high_edge['edge_density']:.4f}, 内部平均相似度{high_edge['mean_sim_internal']:.4f}")
print(f"普通区:   内部边{normal_edge['n_internal_edges']}条, "
      f"边密度{normal_edge['edge_density']:.4f}, 内部平均相似度{normal_edge['mean_sim_internal']:.4f}")

# 6c. 理化指标对比
print("\n--- 6c. 理化指标对比 ---")
phys_cols = ['LogP', 'TPSA', 'MolWt', 'Dipole_Proxy', 'Max_Partial_Charge', 'Balaban_J', 'Bertz_CT']
phys_names = ['LogP(亲脂性)', 'TPSA(极性表面积)', 'MolWt(分子量)',
              'Dipole_Proxy(偶极)', 'Max_Partial_Charge(最大局部电荷)',
              'Balaban_J(拓扑指数)', 'Bertz_CT(结构复杂度)']

phys_comparison = []
for col, name in zip(phys_cols, phys_names):
    h_mean = high_df[col].mean()
    h_std = high_df[col].std()
    n_mean = normal_df[col].mean()
    n_std = normal_df[col].std()
    t_s, p_v = stats.ttest_ind(high_df[col], normal_df[col])
    phys_comparison.append({
        '指标': name, '高活性区均值': f"{h_mean:.3f}±{h_std:.3f}",
        '普通区均值': f"{n_mean:.3f}±{n_std:.3f}",
        '差异': f"{h_mean-n_mean:+.3f}",
        'p值': f"{p_v:.4f}",
        '显著性': '***' if p_v < 0.001 else ('**' if p_v < 0.01 else ('*' if p_v < 0.05 else 'ns'))
    })
    print(f"  {name}: 高活性={h_mean:.3f}, 普通={n_mean:.3f}, p={p_v:.4f}")

phys_comp_df = pd.DataFrame(phys_comparison)

# 图4：理化指标对比箱线图（选取关键4个）
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
plot_cols = ['LogP', 'TPSA', 'MolWt', 'Bertz_CT']
plot_titles = ['LogP (亲脂性)', 'TPSA (极性表面积)', 'MolWt (分子量)', 'Bertz_CT (结构复杂度)']
for ax, col, title in zip(axes.flat, plot_cols, plot_titles):
    data_to_plot = [high_df[col].values, normal_df[col].values]
    bp = ax.boxplot(data_to_plot, labels=['高活性密集区', '普通区域'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#3498db')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
plt.suptitle('关键理化指标区域对比', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'fig4_理化指标对比.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig4 已保存")

# 图5：活性得分对比 + 相似度分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：活性得分小提琴图
data_act = [high_df['Bioactivity_Score'].values, normal_df['Bioactivity_Score'].values]
vp = axes[0].violinplot(data_act, positions=[1, 2], showmeans=True,
                         showmedians=True)
vp['bodies'][0].set_facecolor('#e74c3c')
vp['bodies'][1].set_facecolor('#3498db')
for body in vp['bodies']:
    body.set_alpha(0.6)
axes[0].set_xticks([1, 2])
axes[0].set_xticklabels(['高活性密集区', '普通区域'])
axes[0].set_ylabel('Bioactivity_Score', fontsize=11)
axes[0].set_title('活性得分区域对比', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# 右：内部Tanimoto相似度分布
high_internal_sims = edges[
    edges['Source'].isin(high_ids) & edges['Target'].isin(high_ids)
]['Tanimoto_Similarity']
normal_internal_sims = edges[
    edges['Source'].isin(normal_ids) & edges['Target'].isin(normal_ids)
]['Tanimoto_Similarity']

axes[1].hist(high_internal_sims, bins=20, alpha=0.5, color='#e74c3c',
             label=f"高活性区 (均值={high_internal_sims.mean():.3f})", density=True)
axes[1].hist(normal_internal_sims, bins=20, alpha=0.5, color='#3498db',
             label=f"普通区 (均值={normal_internal_sims.mean():.3f})", density=True)
axes[1].set_xlabel('Tanimoto 相似度', fontsize=11)
axes[1].set_ylabel('密度', fontsize=11)
axes[1].set_title('区域内分子结构相似度分布', fontsize=12)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'fig5_活性与相似度对比.png'), dpi=150)
plt.close()
print("  fig5 已保存")

# ========== 7. 代表分子挑选 ==========
print("\n" + "=" * 60)
print("阶段5：代表分子挑选")
print("=" * 60)

def select_representatives(region_df, edges_df, n_rep=3):
    """从区域中挑选代表性分子：
       标准1：活性得分接近区域中位数（代表性）
       标准2：接近区域空间中心（空间代表性）
       标准3：分子间结构多样性（Tanimoto相似度不太高）
    """
    region_ids = set(region_df['ID'].values)
    coords_r = region_df[['Manifold_X', 'Manifold_Y']].values
    center = coords_r.mean(axis=0)
    median_act = region_df['Bioactivity_Score'].median()

    # 计算每个分子的综合得分
    scores = []
    for i, row in region_df.iterrows():
        mol_id = row['ID']
        # 活性接近度
        act_score = 1.0 - abs(row['Bioactivity_Score'] - median_act) / (region_df['Bioactivity_Score'].std() + 1e-6)
        act_score = max(0, act_score)
        # 空间中心接近度
        dist_to_center = np.sqrt((row['Manifold_X'] - center[0])**2 +
                                 (row['Manifold_Y'] - center[1])**2)
        max_dist = np.sqrt(((coords_r - center)**2).sum(axis=1)).max()
        spatial_score = 1.0 - dist_to_center / (max_dist + 1e-6)
        # 综合得分
        total_score = 0.5 * act_score + 0.5 * spatial_score
        scores.append((mol_id, total_score, act_score, spatial_score))

    scores.sort(key=lambda x: x[1], reverse=True)

    # 从高分到低分选，保证多样性
    selected = []
    selected_sims = []
    for mol_id, total, act_s, spa_s in scores:
        if len(selected) >= n_rep:
            break
        # 检查与已选分子的结构相似度
        too_similar = False
        for sel_id in selected:
            edge1 = edges_df[(edges_df['Source'] == mol_id) & (edges_df['Target'] == sel_id)]
            edge2 = edges_df[(edges_df['Source'] == sel_id) & (edges_df['Target'] == mol_id)]
            sim = pd.concat([edge1, edge2])['Tanimoto_Similarity'].values
            if len(sim) > 0 and sim[0] > 0.6:
                too_similar = True
                break
        if not too_similar:
            selected.append(mol_id)
            selected_sims.append((total, act_s, spa_s))

    return selected, selected_sims

print("\n--- 高活性密集区域代表分子 ---")
high_rep_ids, high_rep_scores = select_representatives(high_df, edges, n_rep=3)
for mid, (total, act_s, spa_s) in zip(high_rep_ids, high_rep_scores):
    row = df[df['ID'] == mid].iloc[0]
    print(f"  分子{mid}: 活性={row['Bioactivity_Score']:.3f}, "
          f"坐标=({row['Manifold_X']:.2f},{row['Manifold_Y']:.2f}), "
          f"综合得分={total:.3f}")

print("\n--- 普通区域代表分子 ---")
normal_rep_ids, normal_rep_scores = select_representatives(normal_df, edges, n_rep=3)
for mid, (total, act_s, spa_s) in zip(normal_rep_ids, normal_rep_scores):
    row = df[df['ID'] == mid].iloc[0]
    print(f"  分子{mid}: 活性={row['Bioactivity_Score']:.3f}, "
          f"坐标=({row['Manifold_X']:.2f},{row['Manifold_Y']:.2f}), "
          f"综合得分={total:.3f}")

# 图6：代表分子在空间中的位置
fig, ax = plt.subplots(figsize=(10, 7))
# 背景所有分子
ax.scatter(df['Manifold_X'], df['Manifold_Y'], c='lightgray', s=15, alpha=0.5)
# 高活性区分子
ax.scatter(high_df['Manifold_X'], high_df['Manifold_Y'],
          c='#e74c3c', s=25, alpha=0.5, label='高活性密集区域')
ax.scatter(normal_df['Manifold_X'], normal_df['Manifold_Y'],
          c='#3498db', s=25, alpha=0.5, label='普通区域')
# 代表分子
for mid in high_rep_ids:
    row = df[df['ID'] == mid].iloc[0]
    ax.scatter(row['Manifold_X'], row['Manifold_Y'], c='darkred', s=120,
              marker='*', edgecolors='white', linewidth=1.5, zorder=5)
    ax.annotate(f"#{mid}", (row['Manifold_X'], row['Manifold_Y']),
               textcoords="offset points", xytext=(10, 10), fontsize=9,
               color='darkred', fontweight='bold')
for mid in normal_rep_ids:
    row = df[df['ID'] == mid].iloc[0]
    ax.scatter(row['Manifold_X'], row['Manifold_Y'], c='darkblue', s=120,
              marker='*', edgecolors='white', linewidth=1.5, zorder=5)
    ax.annotate(f"#{mid}", (row['Manifold_X'], row['Manifold_Y']),
               textcoords="offset points", xytext=(10, 10), fontsize=9,
               color='darkblue', fontweight='bold')

ax.set_xlabel('Manifold_X', fontsize=11)
ax.set_ylabel('Manifold_Y', fontsize=11)
ax.set_title('代表分子空间分布（★星标）', fontsize=12)
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'fig6_代表分子.png'), dpi=150)
plt.close()
print("  fig6 已保存")

# ========== 8. 保存中间结果 ==========
print("\n" + "=" * 60)
print("保存分析结果...")

# 带区域标签的分子数据
df[['ID', 'SMILES', 'Manifold_X', 'Manifold_Y', 'Bioactivity_Score',
    'cluster', 'region']].to_csv(
    os.path.join(CHARTS_DIR, 'molecule_cluster_results.csv'), index=False, encoding='utf-8-sig')

# 理化指标对比表
phys_comp_df.to_csv(os.path.join(CHARTS_DIR, 'phys_comparison.csv'), index=False, encoding='utf-8-sig')

# 簇统计表
cluster_df_stats.to_csv(os.path.join(CHARTS_DIR, 'cluster_stats.csv'), index=False, encoding='utf-8-sig')

print("所有结果已保存到 charts/ 目录")

# 输出汇总，供报告生成使用
print("\n" + "=" * 60)
print("关键结果汇总")
print("=" * 60)
print(f"分子总数: {len(df)}")
print(f"DBSCAN参数: eps={best_eps}, min_samples={min_samples}")
print(f"聚类簇数: {n_clusters}, 噪声点数: {n_noise}")
print(f"高活性密集区域: {len(high_df)}个分子, 平均活性={high_df['Bioactivity_Score'].mean():.3f}")
print(f"普通区域: {len(normal_df)}个分子, 平均活性={normal_df['Bioactivity_Score'].mean():.3f}")
print(f"活性差异t检验: p={p_val:.6f}")
print(f"高活性区代表分子: {high_rep_ids}")
print(f"普通区代表分子: {normal_rep_ids}")

input("\n运行完毕，按回车键退出...")
