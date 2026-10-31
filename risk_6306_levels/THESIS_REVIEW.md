# 6306 固定24次实验：结果复核

已生成的真实文本结果分析：
output_6306_levels/thesis_text_review_20261031/

先阅读 REPORT.md。figures/ 有14张SVG，可在浏览器打开。
metrics_with_combined.csv 保留每个种子，seed_summary.csv 是三种子的均值与范围。
paired_block_bootstrap.csv 保留7/14/28日分块结果、2000次配对重采样及零事件重采样次数。
provenance.json 记录26组源结果文件哈希。
本目录AP沿用原summary.json；逐日事件数、占据单元数、覆盖、面积和日期已经核对。
本目录图中的5/10/20%是三个预先确定的预算点，不是完整面积曲线。

## 完整预测复核、全面积曲线与空间案例

在项目根目录、已激活env2的PowerShell运行：

    python risk_6306_levels\review_6306_thesis.py

脚本只读取数据和已保存预测，不训练，也不加载pth文件。
默认创建 output_6306_levels/thesis_review_时间戳/，已有目录不覆盖。
先核对数据/核心代码指纹、24次配置、标签、日期、网格、AP、覆盖和逐日命中。
任何一项不一致就停止，不能跳过错误继续解释结果。
成功时末行打印 Completed postprocessing，并生成completed.json。

输出：
- REPORT.md：指标、配对区间、按阶段的数值解读和结论限制。
- verification.csv、provenance.json：复核状态、来源哈希和统计设置。
- area_coverage_all_cells.csv、figures/coverage_*：每一个整数网格面积的完整曲线，PNG和PDF。
- paired_block_bootstrap.csv：A/B和合并覆盖差，7/14/28自然日分块。
- daily_hits.csv、positive_cell_ranks.csv：逐日命中、阳性单元排名及同分期望覆盖。
- case_index.csv、cases/：固定seed42；全部高能事件日期，以及每期低能目标相对历史能量首次改善和退步案例。

## 重现文本分析

    python risk_6306_levels\review_6306_tables.py

默认创建独立thesis_text_review_时间戳目录，生成PNG及SVG。
该脚本只依赖summary.json和daily_metrics.csv，不能替代完整预测数组复核。

## 验证

    python risk_6306_levels\test_review_6306.py

5项测试包括所有整数面积与原同分规则一致、自然日缺口、分期配对重采样、
故意错误被拒绝，以及24组模拟文件完整生成报告和图表。
测试使用临时合成数据，不修改真实实验输出。
已在独立环境通过；真实预测数组的完整运行仍需在env2完成。

## 固定解释规则

- 10%仍为原主要预算，5%和20%作补充；不按曲线表现重选主要预算。
- 三种子是三次重复拟合，不是新增事件，也不是平均分数图的集成。
- 合并A/B覆盖先加命中与事件数，再相除；不平均或合并AP。
- 每日固定面积覆盖不能证明知道哪天报警；分数不能直接称为已校准风险概率。
- 分块区间是给定模型和观测目录下的探索性逐项区间，无多重比较校正。
- 高能事件少、全命中造成的零宽区间不能证明等效或总体准确率100%。
- 监测完整性、爆破筛除和工程边界仍需现场资料核实。
