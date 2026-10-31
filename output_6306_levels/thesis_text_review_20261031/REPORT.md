# 6306 固定实验后处理报告

本报告使用24次真实实验和A/B基线的文本输出。逐日事件数、阳性单元数、覆盖率、面积和评价日期已交叉核对；AP沿用原summary.json，尚未从完整预测数组重算。本次未训练网络。

## 核心结果（网络为三种子均值，括号内为范围）

|阶段|能级|方法|AP|5%预算覆盖|10%预算覆盖|
|---|---|---|---|---|---|
|A|ge1e4|Training climatology|0.01198|53.33%|72.33%|
|A|ge1e4|Recent exceedance occupancy|0.09654|78.33%|85.00%|
|A|ge1e4|Recent count|0.06844|93.33%|100.00%|
|A|ge1e4|Recent energy|0.10506|95.00%|100.00%|
|A|ge1e4|Count + energy|0.07961|95.00%|100.00%|
|A|ge1e5|Training climatology|0.00010|0.00%|0.00%|
|A|ge1e5|Recent exceedance occupancy|0.00683|25.00%|25.00%|
|A|ge1e5|Recent count|0.00519|100.00%|100.00%|
|A|ge1e5|Recent energy|0.01766|100.00%|100.00%|
|A|ge1e5|Count + energy|0.00588|100.00%|100.00%|
|B|ge1e4|Training climatology|0.00917|23.68%|69.74%|
|B|ge1e4|Recent exceedance occupancy|0.06546|59.21%|81.58%|
|B|ge1e4|Recent count|0.03704|89.47%|98.68%|
|B|ge1e4|Recent energy|0.04845|89.47%|98.68%|
|B|ge1e4|Count + energy|0.04003|89.47%|98.68%|
|B|ge1e5|Training climatology|0.00017|18.18%|20.00%|
|B|ge1e5|Recent exceedance occupancy|0.00016|18.18%|20.00%|
|B|ge1e5|Recent count|0.00315|80.00%|100.00%|
|B|ge1e5|Recent energy|0.00472|80.00%|100.00%|
|B|ge1e5|Count + energy|0.00345|80.00%|100.00%|
|A|ge1e4|Single low|0.08624 (0.07532–0.09213)|90.56% (90.00–91.67)|100.00% (100.00–100.00)|
|A|ge1e5|Single high|0.02565 (0.01698–0.03094)|100.00% (100.00–100.00)|100.00% (100.00–100.00)|
|A|ge1e4|Dual ordered|0.08733 (0.08257–0.09069)|88.33% (86.67–90.00)|99.44% (98.33–100.00)|
|A|ge1e5|Dual ordered|0.01163 (0.01098–0.01270)|100.00% (100.00–100.00)|100.00% (100.00–100.00)|
|A|ge1e4|Dual shuffled|0.09789 (0.09126–0.10542)|91.11% (90.00–91.67)|97.22% (96.67–98.33)|
|A|ge1e5|Dual shuffled|0.01112 (0.00752–0.01518)|100.00% (100.00–100.00)|100.00% (100.00–100.00)|
|B|ge1e4|Single low|0.08492 (0.08038–0.08843)|92.11% (90.79–94.74)|100.00% (100.00–100.00)|
|B|ge1e5|Single high|0.00919 (0.00574–0.01473)|100.00% (100.00–100.00)|100.00% (100.00–100.00)|
|B|ge1e4|Dual ordered|0.07784 (0.07447–0.08102)|93.42% (93.42–93.42)|99.56% (98.68–100.00)|
|B|ge1e5|Dual ordered|0.06611 (0.02496–0.10281)|100.00% (100.00–100.00)|100.00% (100.00–100.00)|
|B|ge1e4|Dual shuffled|0.07062 (0.06295–0.08314)|93.42% (93.42–93.42)|99.56% (98.68–100.00)|
|B|ge1e5|Dual shuffled|0.08530 (0.00690–0.20540)|93.33% (80.00–100.00)|100.00% (100.00–100.00)|
|A+B|ge1e4|Recent count|不合并AP|91.18%|99.26%|
|A+B|ge1e4|Count + energy|不合并AP|91.91%|99.26%|
|A+B|ge1e4|Recent energy|不合并AP|91.91%|99.26%|
|A+B|ge1e4|Recent exceedance occupancy|不合并AP|67.65%|83.09%|
|A+B|ge1e4|Dual ordered|不合并AP|91.18% (90.44–91.91)|99.51% (98.53–100.00)|
|A+B|ge1e4|Single low|不合并AP|91.42% (90.44–92.65)|100.00% (100.00–100.00)|
|A+B|ge1e4|Dual shuffled|不合并AP|92.40% (91.91–92.65)|98.53% (97.79–99.26)|
|A+B|ge1e4|Training climatology|不合并AP|36.76%|70.88%|
|A+B|ge1e5|Recent count|不合并AP|88.89%|100.00%|
|A+B|ge1e5|Count + energy|不合并AP|88.89%|100.00%|
|A+B|ge1e5|Recent energy|不合并AP|88.89%|100.00%|
|A+B|ge1e5|Recent exceedance occupancy|不合并AP|21.21%|22.22%|
|A+B|ge1e5|Dual ordered|不合并AP|100.00% (100.00–100.00)|100.00% (100.00–100.00)|
|A+B|ge1e5|Single high|不合并AP|100.00% (100.00–100.00)|100.00% (100.00–100.00)|
|A+B|ge1e5|Dual shuffled|不合并AP|96.30% (88.89–100.00)|100.00% (100.00–100.00)|
|A+B|ge1e5|Training climatology|不合并AP|10.10%|11.11%|

## 配对差异（A+B，三种子平均表现，14日分块）

以下为覆盖率差，单位为百分点；不是三个模型平均预测图的表现。5%、20%为补充；10%仍为原主要预算。

|能级|比较|预算|差异|95%区间|无事件重采样次数|
|---|---|---|---|---|---|
|ge1e4|Single low − Recent count|5%|0.25|[-4.09, 4.29]|0|
|ge1e4|Single low − Recent count|10%|0.74|[0.00, 2.67]|0|
|ge1e4|Single low − Recent count|20%|0.00|[0.00, 0.00]|0|
|ge1e4|Single low − Recent energy|5%|-0.49|[-4.07, 2.65]|0|
|ge1e4|Single low − Recent energy|10%|0.74|[0.00, 2.67]|0|
|ge1e4|Single low − Recent energy|20%|0.00|[0.00, 0.00]|0|
|ge1e4|Single low − Count + energy|5%|-0.49|[-5.06, 3.68]|0|
|ge1e4|Single low − Count + energy|10%|0.74|[0.00, 2.67]|0|
|ge1e4|Single low − Count + energy|20%|0.00|[0.00, 0.00]|0|
|ge1e4|Dual ordered − Recent count|5%|0.00|[-4.69, 4.65]|0|
|ge1e4|Dual ordered − Recent count|10%|0.25|[-1.04, 2.32]|0|
|ge1e4|Dual ordered − Recent count|20%|0.00|[0.00, 0.00]|0|
|ge1e4|Dual ordered − Recent energy|5%|-0.74|[-5.61, 3.85]|0|
|ge1e4|Dual ordered − Recent energy|10%|0.25|[-1.04, 2.32]|0|
|ge1e4|Dual ordered − Recent energy|20%|0.00|[0.00, 0.00]|0|
|ge1e4|Dual ordered − Count + energy|5%|-0.74|[-5.56, 3.98]|0|
|ge1e4|Dual ordered − Count + energy|10%|0.25|[-1.04, 2.32]|0|
|ge1e4|Dual ordered − Count + energy|20%|0.00|[0.00, 0.00]|0|
|ge1e4|Dual ordered − Single low|5%|-0.25|[-2.49, 2.03]|0|
|ge1e4|Dual ordered − Single low|10%|-0.49|[-1.24, 0.00]|0|
|ge1e4|Dual ordered − Single low|20%|0.00|[0.00, 0.00]|0|
|ge1e4|Dual ordered − Dual shuffled|5%|-1.23|[-3.40, 0.59]|0|
|ge1e4|Dual ordered − Dual shuffled|10%|0.98|[-0.21, 2.34]|0|
|ge1e4|Dual ordered − Dual shuffled|20%|0.00|[0.00, 0.00]|0|
|ge1e5|Single high − Recent count|5%|11.11|[0.00, 38.54]|1|
|ge1e5|Single high − Recent count|10%|0.00|[0.00, 0.00]|1|
|ge1e5|Single high − Recent count|20%|0.00|[0.00, 0.00]|1|
|ge1e5|Single high − Recent energy|5%|11.11|[0.00, 38.54]|1|
|ge1e5|Single high − Recent energy|10%|0.00|[0.00, 0.00]|1|
|ge1e5|Single high − Recent energy|20%|0.00|[0.00, 0.00]|1|
|ge1e5|Single high − Count + energy|5%|11.11|[0.00, 38.54]|1|
|ge1e5|Single high − Count + energy|10%|0.00|[0.00, 0.00]|1|
|ge1e5|Single high − Count + energy|20%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Recent count|5%|11.11|[0.00, 38.54]|1|
|ge1e5|Dual ordered − Recent count|10%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Recent count|20%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Recent energy|5%|11.11|[0.00, 38.54]|1|
|ge1e5|Dual ordered − Recent energy|10%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Recent energy|20%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Count + energy|5%|11.11|[0.00, 38.54]|1|
|ge1e5|Dual ordered − Count + energy|10%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Count + energy|20%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Single high|5%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Single high|10%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Single high|20%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Dual shuffled|5%|3.70|[0.00, 12.85]|1|
|ge1e5|Dual ordered − Dual shuffled|10%|0.00|[0.00, 0.00]|1|
|ge1e5|Dual ordered − Dual shuffled|20%|0.00|[0.00, 0.00]|1|

## 按阶段计算的结果解读

- A / ge1e4：Single low AP均值 0.08624，历史能量AP 0.10506；三个种子均低于历史能量。
- A / ge1e4：Dual ordered AP均值 0.08733，历史能量AP 0.10506；三个种子均低于历史能量。
- A / ge1e4：正常顺序减打乱顺序的平均AP差为 -0.01055；需结合所有阶段和种子判断，不能由单段差异推断稳定顺序收益。
- B / ge1e4：Single low AP均值 0.08492，历史能量AP 0.04845；三个种子均高于历史能量。
- B / ge1e4：Dual ordered AP均值 0.07784，历史能量AP 0.04845；三个种子均高于历史能量。
- B / ge1e4：正常顺序减打乱顺序的平均AP差为 +0.00721；需结合所有阶段和种子判断，不能由单段差异推断稳定顺序收益。
- A / ge1e5：Single high AP均值 0.02565，历史能量AP 0.01766；不同种子有交叉，不能一致判为高于或低于历史能量。
- A / ge1e5：Dual ordered AP均值 0.01163，历史能量AP 0.01766；三个种子均低于历史能量。
- A / ge1e5：正常顺序减打乱顺序的平均AP差为 +0.00051；需结合所有阶段和种子判断，不能由单段差异推断稳定顺序收益。
- B / ge1e5：Single high AP均值 0.00919，历史能量AP 0.00472；三个种子均高于历史能量。
- B / ge1e5：Dual ordered AP均值 0.06611，历史能量AP 0.00472；三个种子均高于历史能量。
- B / ge1e5：正常顺序减打乱顺序的平均AP差为 -0.01918；需结合所有阶段和种子判断，不能由单段差异推断稳定顺序收益。
- A+B / ge1e4 / 5%：Single low相对历史能量覆盖差 -0.49 个百分点，95%区间[-4.07, 2.65]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e4 / 10%：Single low相对历史能量覆盖差 +0.74 个百分点，95%区间[0.00, 2.67]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e4 / 20%：Single low相对历史能量覆盖差 +0.00 个百分点，95%区间[0.00, 0.00]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e4 / 5%：Dual ordered相对历史能量覆盖差 -0.74 个百分点，95%区间[-5.61, 3.85]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e4 / 10%：Dual ordered相对历史能量覆盖差 +0.25 个百分点，95%区间[-1.04, 2.32]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e4 / 20%：Dual ordered相对历史能量覆盖差 +0.00 个百分点，95%区间[0.00, 0.00]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e5 / 5%：Single high相对历史能量覆盖差 +11.11 个百分点，95%区间[0.00, 38.54]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e5 / 10%：Single high相对历史能量覆盖差 +0.00 个百分点，95%区间[0.00, 0.00]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e5 / 20%：Single high相对历史能量覆盖差 +0.00 个百分点，95%区间[0.00, 0.00]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e5 / 5%：Dual ordered相对历史能量覆盖差 +11.11 个百分点，95%区间[0.00, 38.54]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e5 / 10%：Dual ordered相对历史能量覆盖差 +0.00 个百分点，95%区间[0.00, 0.00]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。
- A+B / ge1e5 / 20%：Dual ordered相对历史能量覆盖差 +0.00 个百分点，95%区间[0.00, 0.00]；区间跨越或接触0，尚不足以支持覆盖率稳定增加。

## 可复核的结论与限制

- 历史热点基线与网络在相同日期、网格和面积规则下比较；具体优势须分别按阶段、能级、指标判断。
- 联合学习是否帮助：查看 Dual ordered 与 Single low/high 的同阶段AP、种子范围和配对覆盖差。
- 时间顺序是否帮助：查看 Dual ordered 与 Dual shuffled；不能把保留历史集合的打乱试验解释为去掉历史信息。
- 本目录只展示原定5/10/20%三个预算点；完整整数网格曲线由另一脚本生成。
- 相同分数采用分数权重，覆盖是期望覆盖；个别案例命中数可能为小数。
- AP对每段全部日期—网格计算。A+B仅按事件数合并覆盖，不平均AP，也不拼接不同阶段模型分数计算AP。
- 不确定性采用7/14/28自然日非重叠分块；14日为主，分阶段重采样后合并A+B。缺测/排除日期不补为正常监测日。
- 区间针对给定训练模型、已观测目录的配对覆盖差；不包含重新训练的不确定性。三种子展示重复拟合波动，不能扩大独立事件数。
- 区间为探索性逐项百分位区间，未作多重比较校正；高能事件少或全命中时区间可退化为0，不能据此证明等效或总体完美预测。
- 固定每日面积不能证明时间预警有效。概率校准、监测完整性、爆破筛除与工程边界仍需分别核实。
- 本目录尚无空间案例图；完整脚本固定seed42生成全部高能日期和低阈值首次改善/退步案例。

## 文件导航（本目录为文本结果分析）

- verification.csv：文本结果一致性核对；provenance.json：来源哈希和统计设置。
- metrics_with_combined.csv / seed_summary.csv：各阶段、能级、方法与种子指标。
- paired_block_bootstrap.csv：7/14/28日分块的配对覆盖差及区间。
- daily_hits.csv：所有方法逐日命中，保留事件数和种子。
- figures/：AP散点、5/10/20%三个预算点、配对覆盖差。三个预算点不代表完整面积曲线。
- 完整曲线、空间案例和预测数组复核：运行 review_6306_thesis.py，在独立thesis_review_时间戳目录生成。
