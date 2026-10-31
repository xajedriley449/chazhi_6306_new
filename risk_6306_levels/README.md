# 6306 分能级空间超限预测

独立实验目录；Python >= 3.10。保留原 risk_6306 数据、脚本和结果。
从项目根目录、已激活的 env2 运行。已有 CUDA 版 PyTorch 时沿用现有环境。

## 固定的研究定义

输入过去10个完整自然日，预测下一自然日每个固定空间单元：
- ge1e4：是否至少发生一次 E >= 10000 J 的事件（主要建模目标）。
- ge1e5：是否至少发生一次 E >= 100000 J 的事件（高能目标，单独评价）。

标签顺序固定为 [ge1e4, ge1e5]，数组 [day, level, y, x]。
同一个高能事件同时计入两个累计目标，不能把两个能级的事件数相加。
它们不是互斥能级分类；较低阈值的成功不证明高能矿震预测成功。

保留三个输入通道：[全部事件频次密度、E<1e5能量密度、E>=1e5能量密度]。
新增1e4输出并不要求重分输入能量通道。本轮保留输入是为了控制改动。
平滑只用于历史输入；输出是坐标落格的硬二值标签。约100m单元、50m输入带宽是候选实验参数。

双输出模型用 p4=sigmoid(a)，p5=p4*sigmoid(b)，保证 p5<=p4。
使用两个边际输出的未加权BCE等权平均，不进行正例过采样，不按全数据拟合归一化。
输出是尚未验证校准的概率估计，不能直接作为工程预警概率。

## 文件

- prepare_6306_levels.py：从已有清洗事件表、日清单、空间配置构造新数据。
- run_6306_levels.py：基线、固定轮次训练、保存模型评价、指定日期预测。
- model.py：单层ConvLSTM与嵌套输出。
- common.py：数据完整性校验、训练期缩放、统一面积评价。
- test_core.py：独立数据/评价测试；PyTorch存在时加测输出、反传和参数重载。

## 第一轮只执行以下步骤

```powershell
cd E:\bye\code\chazhi_6306_new
python -m pip install -r risk_6306_levels\requirements.txt
python risk_6306_levels\test_core.py
python risk_6306_levels\prepare_6306_levels.py --reuse-samples input_6306_spatial_risk/sample_manifest.csv
python risk_6306_levels\run_6306_levels.py baseline --output output_6306_levels/baselines_development
python risk_6306_levels\run_6306_levels.py train --task dual --seed 42 --figures
python risk_6306_levels\run_6306_levels.py train --task high --seed 42
python risk_6306_levels\run_6306_levels.py train --task low --seed 42
```

新数据默认 input_6306_levels；复用样本清单可确保与旧实验采用相同可用日期。
输出目录非空时拒绝覆盖。重跑必须显式指定新的 --output 路径。
三个模型均固定15轮、相同输入和seed。low/high用于检查多任务有无收益或负迁移。
不根据后续评价损失挑选轮次，不早停、不挑最优分数报警阈值。
默认 development 用原 train 训练、原 val 评价，不自动评价原 test。
本轮最多3次正式训练，完成后先总结，不因为结果不好扫描新阈值或增加网络模块。
不要直接将旧模型与本模型的数字作为纯消融：旧训练选模和报警面积规则不同。

## 输出与比较

例如双输出结果：output_6306_levels/train_development_dual_all_seed42/
- run_config.json：数据与代码哈希、设置、实际训练和评价日期。
- training_summary.json、loss_log.csv：固定轮次、训练损失、训练期标签比例。
- last_model.pth：最后一轮模型；没有 best_model.pth。
- evaluation/summary.json：每个能级各方法的AP、Brier、log loss、5/10/20%面积覆盖。
- evaluation/daily_metrics.csv：包含无目标日期的逐日覆盖和报警面积。
- evaluation/positive_cell_diagnostics.csv：逐阳性单元诊断；不是逐事件ID表。
- evaluation/predictions.npz：分能级预测/标签/事件数/日期/网格边界。
- evaluation/reliability.csv：校准分箱诊断；不用于自动校准。
- --figures 保存最多10个示例日/能级的预测图，青色叉为观测阳性单元中心。

所有方法每日使用严格相同的 floor(网格数*预算) 格面积。
并列分数（包括零）采用均匀随机打破并列的期望权重，所以命中事件数可能是小数。
这不是实际执行的二值报警区；不要把期望覆盖写成确切命中次数。
固定面积评价衡量空间排序，不证明能判断哪天应报警。
主比较固定10%面积覆盖，5%、20%、全时空AP和Brier作补充。
无目标事件的评价段仍输出结果，AP/事件召回为null，不伪造为0或1。

## 后续可用的时间推进回测（首轮不要自动全部跑）

--period A：训练目标日期 <2023-01-01，评价2023年上半年。
--period B：训练目标日期 <2023-07-01，评价2023年下半年到目录末尾。
输入都严格早于目标日；B可以使用A中已经发生的观测，A/B评价日期不重叠。
历史数据已被查看，必须称为历史回测，不能称完全未见独立测试。
固定15轮，不用A成绩修改B设置。可预先固定dual/high、seed42/43/44、A/B共12次；
此批次只有在首轮结束并固定方案后才执行，不追加阈值、种子或网络。
两段召回按命中事件数与事件总数合并，不直接平均百分比；AP应合并预测后计算。
多个种子不是新增独立观测，不能把事件数量乘以3。

单次示例：
```powershell
python risk_6306_levels\run_6306_levels.py train --period A --task dual --seed 42
```

## 重读保存模型与预测

```powershell
python risk_6306_levels\run_6306_levels.py evaluate --checkpoint output_6306_levels/train_development_dual_all_seed42/last_model.pth --output output_6306_levels/review_dual42 --figures
python risk_6306_levels\run_6306_levels.py forecast --checkpoint output_6306_levels/train_development_dual_all_seed42/last_model.pth --forecast-date 2023-06-01 --output output_6306_levels/forecast_20230601
```

指定日期预测只读该日前10天输入，日期必须晚于模型所有训练目标。
此示例是历史日期预测；有已存真值也不用于forecast。
完整性校验绑定原数据和实现。当前不提供将新日目录追加后直接复用旧checkpoint的在线更新流程。
模型只加载自己生成的checkpoint。更新代码/数据后需重训，避免悄悄改变实验含义。

## 数据边界与解释限制

默认复用原 exploratory 口径：unknown监测状态的空目录日被视作探索性的零事件日，
不代表已确认正常监测。missing日、越界事件日及其输入窗、未确认目录首末日继续排除。
--coverage strict 仅使用确认正常监测日期，可能没有足够样本，需先补监测状态。
source中的日期、event_id、计数、能量、边界会校验。缩放仅从训练输入历史拟合。
1e4是研究分层，不是经标定的矿震/灾害物理分界；两个阈值都不是冲击地压标签。

## 验证

本次交付的实际检查结果见聊天说明；test_core.py中PyTorch缺失时会明确SKIP模型测试。
通过数据测试不等于通过真实模型训练。先运行上面的自检与准备，再进行三次正式训练。
