# -*- coding: utf-8 -*-
"""
B题 问题1 完整解题报告生成器
数学建模 B 题 问题1：DBSCAN 聚类与活性加权区域识别
"""
import os, sys
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

WORKSPACE = os.environ.get(
    "B1_WORKSPACE",
    os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(WORKSPACE, "charts")
REPORTS_DIR = os.path.join(WORKSPACE, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

doc = Document()

# ===== 样式设置 =====
style = doc.styles['Normal']
font = style.font
font.name = '宋体'
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
font.size = Pt(12)

# ===== 封面 =====
for _ in range(4):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('MIMAC分子数据的聚类筛选与结构探索')
run.font.size = Pt(22)
run.font.bold = True
run.font.color.rgb = RGBColor(0x1a, 0x47, 0x8a)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('—— 问题1：DBSCAN空间聚类与活性加权区域识别')
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_paragraph()

info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = info.add_run('MIMAC分子数据的聚类筛选与结构探索')
run.font.size = Pt(12)

doc.add_page_break()

# ===== 1. 问题重述 =====
h = doc.add_heading('一、问题重述', level=1)
doc.add_paragraph(
    '在药物研发中，研究者需要从大量候选分子中筛选出具有高生物活性的分子进行进一步实验。'
    'MIMAC数据集提供了325个分子在二维流形空间中的坐标、生物活性得分、多种理化指标，'
    '以及分子间的kNN结构相似图。'
)
doc.add_paragraph(
    '问题1要求：建立数学模型对分子在二维流形空间中的分布进行分析，'
    '识别"高活性密集区域"和"普通区域"。需要说明聚类依据，'
    '比较不同区域在活性得分、相似关系、理化指标上的差异，'
    '并挑选代表性分子说明合理性。'
)
doc.add_paragraph(
    '关键约束：不得直接使用无解释的机器学习模型作为黑箱模型；'
    '需遵守《全国大学生数学建模竞赛人工智能工具使用规定（2025）》的要求。'
)

# ===== 2. 数据概览 =====
h = doc.add_heading('二、数据概览', level=1)
doc.add_paragraph(
    '附件1包含325个分子的12个变量：ID、SMILES字符串、Manifold_X/Manifold_Y（二维空间坐标）、'
    'Bioactivity_Score（生物活性得分）、LogP（亲脂性）、TPSA（极性表面积）、'
    'MolWt（分子量）、Dipole_Proxy（偶极代理指标）、Max_Partial_Charge（最大局部电荷）、'
    'Balaban_J（拓扑结构指数）、Bertz_CT（结构复杂度指数）。'
)
doc.add_paragraph(
    '附件2包含1625条kNN图的边（Source→Target, Tanimoto_Similarity），'
    '覆盖325个源节点和307个目标节点。'
)

# 数据统计表
table = doc.add_table(rows=9, cols=4, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ['变量', '均值', '标准差', '范围']
for i, h_text in enumerate(headers):
    table.rows[0].cells[i].text = h_text
data_rows = [
    ('Bioactivity_Score', '4.646', '1.848', '0.00 ~ 10.00'),
    ('Manifold_X', '6.524', '8.266', '-11.96 ~ 23.21'),
    ('Manifold_Y', '6.311', '5.822', '-10.29 ~ 20.94'),
    ('LogP', '1.442', '1.071', '-1.82 ~ 3.59'),
    ('TPSA', '28.995', '22.671', '0.00 ~ 103.76'),
    ('MolWt', '146.027', '28.234', '88.05 ~ 215.03'),
    ('Balaban_J', '2.722', '0.318', '2.012 ~ 3.278'),
    ('Bertz_CT', '225.100', '118.360', '38.81 ~ 587.29'),
]
for i, (var, mean, std, rng) in enumerate(data_rows):
    for j, val in enumerate([var, mean, std, rng]):
        table.rows[i+1].cells[j].text = val

doc.add_paragraph()
doc.add_picture(os.path.join(CHARTS_DIR, 'fig1_空间分布与活性分布.png'),
                width=Inches(5.8))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph('图1  分子二维流形空间分布（左：按活性着色）与活性得分直方图（右）')
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.runs[0].font.size = Pt(10)

# ===== 3. 数学模型 =====
doc.add_page_break()
h = doc.add_heading('三、数学模型与方法', level=1)

h2 = doc.add_heading('3.1 DBSCAN密度聚类', level=2)
doc.add_paragraph(
    'DBSCAN（Density-Based Spatial Clustering of Applications with Noise）'
    '是一种基于密度的聚类算法，其核心思想是：将高密度区域中相互连通的点归为同一簇，'
    '低密度区域的点归为噪声。该算法有两个关键参数：'
)
doc.add_paragraph('（1）eps (ε)：邻域半径。对任意点 p，其 ε-邻域定义为 Nε(p) = {q | dist(p, q) ≤ ε}。')
doc.add_paragraph('（2）min_samples (MinPts)：核心点的最小邻域点数。若 |Nε(p)| ≥ MinPts，则 p 为核心点。')
doc.add_paragraph(
    'DBSCAN的簇定义为：从一个核心点出发，通过密度可达关系（density-reachable）'
    '连通的最大点集。密度可达关系定义为：若存在序列 p1, p2, ..., pn，'
    '其中 p1 = p, pn = q，且对每个 pi，pi+1 是从 pi 直接密度可达的（即 pi+1 ∈ Nε(pi) 且 pi 是核心点），'
    '则称 q 从 p 密度可达。'
)
doc.add_paragraph(
    '选择DBSCAN的理由：（1）它不需要预先指定聚类簇数K，由数据本身的密度分布决定；'
    '（2）它可以发现任意形状的簇，适合流形空间中的分子分布；'
    '（3）它对噪声点具有鲁棒性；'
    '（4）算法原理清晰、可解释，符合竞赛要求。'
)

h2 = doc.add_heading('3.2 eps参数选择：k-距离图法', level=2)
doc.add_paragraph(
    '采用k-距离图（k-distance graph）确定最优eps值。'
    '对每个数据点，计算其到第k近邻的距离（k = min_samples - 1 = 4），'
    '将所有点的k-距离按降序排列并绘图。曲线的"拐点"（elbow point）即为推荐的eps值。'
    '拐点之前距离变化平缓，拐点之后距离急剧增大，标志着从"簇内"过渡到"噪声"。'
)
doc.add_paragraph(
    '从k-距离图（图2）观察，曲线在eps≈1.5处出现明显拐点。'
    '进一步测试了eps=0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5等多个取值：'
    'eps=0.8~1.2时产生15个簇（过细分），eps=1.5时合并为13个簇（合理），'
    'eps≥1.8后稳定在12个簇。综合考虑簇内凝聚度和簇间分离度，选择eps=1.5。'
)

doc.add_picture(os.path.join(CHARTS_DIR, 'fig2_k距离图.png'),
                width=Inches(5.5))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph('图2  k-距离图（k=4），红色虚线标注eps=1.5的拐点位置')
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.runs[0].font.size = Pt(10)

h2 = doc.add_heading('3.3 活性加权区域识别', level=2)
doc.add_paragraph(
    '聚类完成后，计算每个簇的平均生物活性得分。定义：'
)
doc.add_paragraph(
    '"高活性密集区域"：满足两个条件的簇——（1）簇平均活性 ≥ 全局均值 + 0.3×全局标准差；'
    '（2）簇规模 ≥ 5个分子（排除过小的偶然聚集）。'
)
doc.add_paragraph(
    '"普通区域"：不满足上述条件的所有其他簇。'
)
doc.add_paragraph(
    f'依据数据：全局活性均值=4.646，标准差=1.848，高活性阈值=5.201。'
    f'最终识别出3个高活性密集区域簇，共30个分子；其余10个簇为普通区域，共295个分子。'
)

# ===== 4. 结果与分析 =====
doc.add_page_break()
h = doc.add_heading('四、结果与分析', level=1)

h2 = doc.add_heading('4.1 聚类结果可视化', level=2)
doc.add_paragraph(
    '图3展示了DBSCAN聚类结果（左）与活性加权后的区域分类（右）。'
    '可以看到，高活性密集区域（红色）主要集中在流形空间的下方（低Manifold_Y区域）'
    '和左上角（低Manifold_X、中等Manifold_Y区域），'
    '这些区域分子密集且活性得分整体较高。'
)

doc.add_picture(os.path.join(CHARTS_DIR, 'fig3_聚类结果.png'),
                width=Inches(5.8))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph('图3  左：DBSCAN空间聚类结果（13簇）；右：活性加权区域分类（高活性密集区 vs 普通区）')
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.runs[0].font.size = Pt(10)

h2 = doc.add_heading('4.2 活性得分对比', level=2)
doc.add_paragraph(
    f'高活性密集区域的生物活性得分均值=5.326±1.289（n=30），'
    f'普通区域的活性得分均值=4.577±1.883（n=295）。'
    f'独立样本t检验结果：t=2.124, p=0.034<0.05，差异有统计学显著性。'
    f'这说明两类区域确实在生物活性上存在本质差异，验证了活性加权分类的合理性。'
)

h2 = doc.add_heading('4.3 相似关系对比', level=2)
doc.add_paragraph(
    f'从kNN图（附件2）中提取各区域的内部边进行分析：'
)
doc.add_paragraph(
    f'高活性密集区域：内部边150条，边密度=0.345（150/(30×29/2)），'
    f'内部平均Tanimoto相似度=0.596。'
)
doc.add_paragraph(
    f'普通区域：内部边1474条，边密度=0.034（1474/(295×294/2)），'
    f'内部平均Tanimoto相似度=0.516。'
)
doc.add_paragraph(
    '高活性密集区域的边密度（0.345 vs 0.034）和内部平均相似度（0.596 vs 0.516）'
    '均显著高于普通区域。这表明高活性分子在结构上更加聚集和同质化，'
    '而普通区域的分子结构更为分散和多样化。'
    '这一发现符合药物化学的直觉：高活性分子往往共享关键的药效团结构特征。'
)

h2 = doc.add_heading('4.4 理化指标对比', level=2)

# 理化指标对比表
table = doc.add_table(rows=8, cols=5, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ['理化指标', '高活性密集区', '普通区域', '差异', '显著性']
for i, h_text in enumerate(headers):
    table.rows[0].cells[i].text = h_text

phys_data = [
    ('LogP（亲脂性）', '1.247±0.899', '1.462±1.087', '-0.215', 'ns (p=0.48)'),
    ('TPSA（极性表面积）', '36.638±22.701', '28.215±22.598', '+8.423*', 'p=0.035'),
    ('MolWt（分子量）', '159.30±18.52', '144.67±28.49', '+14.63**', 'p=0.004'),
    ('Dipole_Proxy（偶极）', '0.444±0.217', '0.489±0.196', '-0.045', 'ns (p=0.27)'),
    ('Max_Partial_Charge', '0.144±0.107', '0.131±0.119', '+0.013', 'ns (p=0.50)'),
    ('Balaban_J（拓扑指数）', '2.838±0.160', '2.710±0.328', '+0.128*', 'p=0.037'),
    ('Bertz_CT（结构复杂度）', '334.06±89.06', '214.02±114.31', '+120.04***', 'p<0.001'),
]
for i, (name, high, normal, diff, sig) in enumerate(phys_data):
    for j, val in enumerate([name, high, normal, diff, sig]):
        table.rows[i+1].cells[j].text = val

doc.add_paragraph()
doc.add_paragraph(
    '表1显示，高活性密集区域的分子具有更高的极性表面积（TPSA, p=0.035）、'
    '更大的分子量（MolWt, p=0.004）、更复杂的拓扑结构（Balaban_J, p=0.037）'
    '和显著更高的结构复杂度（Bertz_CT, p<0.001）。'
    '这意味着高活性分子倾向于具有更复杂的环系结构和更多的极性官能团，'
    '这可能与其与生物靶标的多点结合能力有关。'
)

doc.add_picture(os.path.join(CHARTS_DIR, 'fig4_理化指标对比.png'),
                width=Inches(5.5))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph('图4  关键理化指标（LogP、TPSA、MolWt、Bertz_CT）区域对比箱线图')
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.runs[0].font.size = Pt(10)

doc.add_picture(os.path.join(CHARTS_DIR, 'fig5_活性与相似度对比.png'),
                width=Inches(5.5))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph('图5  左：活性得分小提琴图；右：区域内Tanimoto相似度分布')
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.runs[0].font.size = Pt(10)

# ===== 5. 代表分子 =====
doc.add_page_break()
h = doc.add_heading('五、代表分子挑选', level=1)

doc.add_paragraph(
    '从每个区域挑选3个代表性分子，挑选标准为：'
)
doc.add_paragraph('（1）活性代表性：生物活性得分接近本区域中位数，代表区域的典型活性水平；')
doc.add_paragraph('（2）空间代表性：在流形空间中的位置接近区域几何中心；')
doc.add_paragraph('（3）结构多样性：与已选分子间的Tanimoto相似度≤0.6，避免选到结构过于相似的冗余分子。')
doc.add_paragraph(
    '综合得分 = 0.5 × 活性接近度 + 0.5 × 空间中心接近度，从高到低排序，'
    '同时满足结构多样性约束。'
)

doc.add_paragraph()

# 高活性区代表分子
p = doc.add_paragraph()
run = p.add_run('高活性密集区域代表分子：')
run.font.bold = True

table = doc.add_table(rows=4, cols=6, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h_text in enumerate(['分子编号', '活性得分', 'Manifold_X', 'Manifold_Y', 'SMILES（截断）', '选取理由']):
    table.rows[0].cells[i].text = h_text

high_reps = [
    ('#117', '4.882', '-7.54', '-3.25', 'CCCn1cnc2cncnc21', '接近区域中心，活性接近中位数'),
    ('#82', '4.952', '-11.85', '0.01', 'COc1nc2cncnc2n1C', '空间中心代表，结构含甲氧基特征'),
    ('#6', '4.803', '-11.92', '0.00', 'Cn1c(Br)nc2cncnc21', '含Br取代基，结构多样性强'),
]
for i, (mid, act, mx, my, smi, reason) in enumerate(high_reps):
    for j, val in enumerate([mid, act, mx, my, smi, reason]):
        table.rows[i+1].cells[j].text = val

doc.add_paragraph()

# 普通区代表分子
p = doc.add_paragraph()
run = p.add_run('普通区域代表分子：')
run.font.bold = True

table = doc.add_table(rows=4, cols=6, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h_text in enumerate(['分子编号', '活性得分', 'Manifold_X', 'Manifold_Y', 'SMILES（截断）', '选取理由']):
    table.rows[0].cells[i].text = h_text

normal_reps = [
    ('#185', '4.489', '6.95', '5.06', 'COc1cccc(CC(=O)O)c1', '综合得分最高，活性代表性强'),
    ('#48', '4.558', '6.78', '6.41', 'OCc1ccc(O)cc1', '双层含氧官能团，结构简单'),
    ('#65', '4.445', '6.57', '5.16', 'COc1ccc(CC(=O)O)cc1', '含羧基特征，与#185不同取代位点'),
]
for i, (mid, act, mx, my, smi, reason) in enumerate(normal_reps):
    for j, val in enumerate([mid, act, mx, my, smi, reason]):
        table.rows[i+1].cells[j].text = val

doc.add_paragraph()

doc.add_picture(os.path.join(CHARTS_DIR, 'fig6_代表分子.png'),
                width=Inches(5.2))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph('图6  代表分子在流形空间中的位置（★星标），红色=高活性区，蓝色=普通区')
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.runs[0].font.size = Pt(10)

doc.add_paragraph(
    '从空间分布来看，高活性区的3个代表分子（#117, #82, #6）'
    '全部位于Manifold_X<0的低纬度区域，具有典型的含氮杂环骨架。'
    '普通区的3个代表分子（#185, #48, #65）位于Manifold_X=6~7的高密度区域，'
    '结构以含氧芳香化合物为主，代表了普通区域分子结构的多样性特征。'
)

# ===== 6. 结论 =====
doc.add_page_break()
h = doc.add_heading('六、结论', level=1)

doc.add_paragraph(
    '1. 采用DBSCAN密度聚类算法（eps=1.5, min_samples=5），在二维流形空间中将325个分子划分为13个空间簇，'
    '无噪声点，聚类结构清晰。eps参数通过k-距离图拐点法确定，避免了主观调参。'
)
doc.add_paragraph(
    '2. 通过活性加权（阈值=全局均值+0.3σ），识别出3个"高活性密集区域"簇（共30个分子），'
    '活性均值=5.326，显著高于其余295个普通区域分子的均值4.577（t检验p=0.034）。'
)
doc.add_paragraph(
    '3. 两类区域在多个维度上存在显著差异——高活性区分子具有更高的Tanimoto结构相似度（0.596 vs 0.516）、'
    '更大的极性表面积（TPSA，p=0.035）、更大的分子量（MolWt，p=0.004）'
    '和显著更高的结构复杂度（Bertz_CT，p<0.001）。'
    '这些特征说明高活性分子倾向于具有更复杂的杂环结构和更多的极性官能团。'
)
doc.add_paragraph(
    '4. 基于活性代表性、空间中心性和结构多样性三个标准，分别为每个区域挑选了3个代表性分子，'
    '可为进一步实验验证提供候选。'
)

# ===== 7. 附录 =====
doc.add_page_break()
h = doc.add_heading('附录：代码与工具声明', level=1)
doc.add_paragraph(
    '分析工具：Python 3.14 + pandas + numpy + scikit-learn（DBSCAN, NearestNeighbors）+ scipy + matplotlib。'
)
doc.add_paragraph(
    'DBSCAN算法为scikit-learn的确定性实现，参数选择基于k-距离图几何分析，'
    '不涉及随机初始化，结果可完全复现。'
)
doc.add_paragraph(
    '所有代码见 code/B题1_DBSCAN聚类分析.py，中间结果见 charts/ 目录。'
)
doc.add_paragraph(
    '本报告遵守《全国大学生数学建模竞赛人工智能工具使用规定（2025）》：'
    '数学模型由人工选定并解释，AI仅辅助代码编写与图表生成。'
)

# ===== 保存 =====
output_path = os.path.join(REPORTS_DIR, 'B题1_DBSCAN聚类分析报告.docx')
doc.save(output_path)
print(f"报告已保存: {output_path}")
