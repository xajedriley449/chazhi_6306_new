# 6306：频次与能量联合输入的高能事件空间预测

将整个 `risk_6306` 文件夹放在：

```text
E:\bye\code\chazhi_6306_new\risk_6306\
    common.py
    model.py
    prepare_6306_spatial_risk.py
    run_6306_spatial_risk.py
    test_core.py
    requirements.txt
    README.md
    validation_reference\
```

这是独立新实验，使用仓库已有的清洗事件、日清单和空间范围。入口脚本从自己的位置定位项目根目录，不依赖当前终端目录。建议仍在项目根目录运行下列命令。

## 1. 环境与源数据

在已激活 `env2` 的 VSCode PowerShell 中运行：

```powershell
cd E:\bye\code\chazhi_6306_new
python -m pip install -r risk_6306\requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

复用以下三个已有文件。新脚本不需要读取旧 PNG，也不需要读取旧64×64日NPZ：

```text
input_6306_daily_xy\events_selected.csv
input_6306_daily_xy\daily_manifest.csv
input_6306_daily_xy\dataset_config.json
```

若这些源文件尚未生成，在项目根目录使用仓库已有预处理脚本：

```powershell
python chazhi_txyz_ST-KDE.py --input data/6306_all.csv --output input_6306_daily_xy --png-every 0
```

该命令仅用于源目录不存在时。源目录已存在就直接进入下一步。已有源目录的日期范围、阈值和空间边界会在新实验配置中保留追溯记录。

核对依据：本包的 `validation_reference/source_revision.json` 保存实际从GitHub `main`读取到的文件SHA。训练接口参考 `run_convlstm-early/pred_6306_xy_v4.py`，新模型使用独立实现和独立检查点。

## 2. 构建新输入和标签

```powershell
python risk_6306\prepare_6306_spatial_risk.py
```

默认输出：`input_6306_spatial_risk`。若要严格沿用指定旧实验的可用样本日期，可以在首次准备时使用：

```powershell
python risk_6306\prepare_6306_spatial_risk.py --reuse-samples output_6306_v3_spatial_final_check/sample_manifest.csv
```

两条准备命令选择一条执行。若旧清单不存在，使用第一条。输出目录非空时脚本拒绝覆盖；重跑请指定新的 `--output`，后续命令相应使用 `--data` 指向它。

每个历史日三个输入通道均位于同一坐标范围：

| 通道 | 内容 | 变换前单位 |
|---|---|---|
| 0 | 全部目录事件频次密度 | 次/公顷 |
| 1 | 单次能量小于10^5 J事件的能量密度 | J/公顷 |
| 2 | 单次能量至少10^5 J事件的能量密度 | J/公顷 |

高斯核仅处理历史输入。对每个空间格积分高斯核后，在固定矩形范围内逐事件归一化；边界事件仍保持计数和能量总量。该方法是空间表示方法，不模拟应力传播。空间分布与原64×64中心采样截断高斯不完全相同，是本新实验明确声明的变化。

预测标签：**目标自然日，该固定空间格内是否至少有一次能量达到10^5 J的事件**。有则1，无则0。标签不平滑、不按能量加权，也不由旧能量图归一化生成。输入为前10个完整自然日，输出为第11天，绝不把目标日输入模型。一天同一格有多次事件，训练标签仍为1，评价同时保留真实事件数量。

默认 `--cell-m 100 --sigma-m 50`，两者都是候选试验值，需要结合定位误差和工程尺度确定。它们不表示已经验证的危险半径或定位精度。范围沿用旧配置；为保持完整范围，每方向均匀分格且实际边长不超过100米。上传原始数据按旧默认流程生成时，网格为13×19，单格约98.54×92.93米。原64×64表示的精细图回归与本实验粗区域占据预测，不应直接比较像素指标。

训练前对输入使用 `log1p`，然后按各通道在**实际训练样本历史日期**中的最大值缩放。验证和测试不重新拟合尺度，不逐帧Min-Max；出现超过1的输入是允许的。

默认时间划分按目标日：训练早于2022-01-01，验证为2022年，测试从2023-01-01开始。首末目录日监测未确认、历史或目标含网格外事件、明确missing日等样本会被排除。完整日历保留，历史跨越集合边界是允许的，因为只能使用目标日以前已经发生的信息。

默认 `--coverage exploratory` 与现有探索实验口径一致，unknown日期按事件目录回测，**无目录记录不等于已确认正常监测下的零事件**。有完整监测记录时，先让旧预处理脚本以 `--coverage monitoring.csv` 生成带状态的源目录，再以本脚本 `--coverage strict` 重建新实验。默认数据全部unknown，strict模式将无可用样本，这是预期行为。

关键文件：`split_summary.json` 是样本统计，`daily_audit.csv` 是逐日检查，`sample_manifest.csv` 是逐样本输入边界、划分和排除原因，`config.json` 记录参数及数据指纹。

## 3. 先运行无需PyTorch的基线

```powershell
python risk_6306\run_6306_spatial_risk.py baseline --output output_6306_spatial_risk/baselines
```

输出验证期的：训练期空间占据频率、前10天高能占据、前10天频次、前10天累计能量、频次与能量等权评分。后三者是排序分数，不称为概率；前两者是经验发生频率估计。

## 4. 训练三通道联合模型

```powershell
python risk_6306\run_6306_spatial_risk.py train --input-mode all --seed 42
```

输出：`output_6306_spatial_risk/all_seed42`。

一个ConvLSTM共享处理3通道历史数据，输出每个区域的一个logit。使用自然样本分布、未加权BCE和AdamW。验证BCE最低的epoch保存为 `best_model.pth`，早停默认15轮，最多100轮。模型不是原能量回归模型，不能载入原模型权重。训练不自动评价测试集。默认自动使用可用CUDA；可指定 `--device cpu`。

模型初始化输出接近训练期正例比例，以免背景数量大导致初始输出过高。未使用正例重采样或类别加权，以保留自然发生频率下BCE的含义；这不能保证少数高能事件一定可学，也不能保证sigmoid已校准。需看验证成绩是否超过基线。若主要输出固定图或全低分，应如实记录，不把低BCE单独作为预测成功证据。

## 5. 对照：只输入频次、只输入能量

```powershell
python risk_6306\run_6306_spatial_risk.py train --input-mode count --seed 42
python risk_6306\run_6306_spatial_risk.py train --input-mode energy --seed 42
```

输出分别为 `count_seed42` 和 `energy_seed42`。三个配置的目标、时间划分、区域尺度、隐藏宽度、训练规则一致，输入通道数不同导致参数量略有差别。它们是输入消融，训练三份模型用于比较，不是把三个独立模型的图拼成最终风险图。

优先检查首个种子是否能超过简单基线。需要检验稳定性时，对相同三种配置预先固定增加42、43、44等种子，避免只挑最好一次。不要基于测试结果选通道、网格、带宽或阈值。

## 6. 阅读结果

每个训练目录包含：

```text
best_model.pth / last_model.pth
loss_log.csv / training_summary.json
run_config.json / thresholds.json / completed.json
val/summary.json
val/daily_metrics.csv
val/frozen_threshold_metrics.json
val/reliability.csv
val/predictions.npz
val/figures/*.png
```

先比较 `val/summary.json` 中同方法的 `top10.event_recall`、`top10.mean_alarm_area` 与基线，再看AP、Brier和log loss。对只有6个高能日期的验证集，一次命中就是很大变化，不能把所有网格看成独立高能样本。零事件日期也参与报警面积和概率误差统计。

`top5/top10/top20` 是每天固定面积预算，分数并列采用分数报警权重，报告随机均匀打破并列时的期望覆盖率；不是声称在并列区内准确定位。全零评分图不报警。由于取整数格，实际面积可能略低于预算。每天固定面积报警的结果主要衡量空间排序，不能单独证明时间预警有效。

另有一套**跨日期共用的固定分数阈值**：仅在验证集选取，在平均面积时间预算不超过10%的约束下尽量提高验证正格召回，再冻结到后续评价。验证阈值指标存在选择乐观偏差，不是独立验证结果。`--max-alarm-area` 只用于基线／训练时确定阈值；测试评价不重新选择。若并列导致无法在预算内报警，可以选择零报警，结果不会强行凑成功。

模型输出以sigmoid发生评分保存。`reliability.csv`、Brier和log loss提供概率评价信息，但未实施额外概率校准。极稀疏目录的校准证据有限，论文优先称高能事件空间发生评分，不能直接当成冲击地压概率、人员伤亡概率或通用矿震危险等级。

## 7. 单独评价历史测试期

固定方案后执行：

```powershell
python risk_6306\run_6306_spatial_risk.py evaluate --checkpoint output_6306_spatial_risk/all_seed42/best_model.pth --split test --output output_6306_spatial_risk/all_seed42_test
```

检查点保存了验证期阈值、输入通道、尺度和数据指纹。评价必须使用相同数据和配置，不能悄悄更换标签或数据集。测试目录会明确标记这段历史数据曾被查看，属于描述性复核。`last_model.pth` 不含最终冻结阈值，请使用 `best_model.pth`。

## 8. 指定未来日预测

接口：`forecast --checkpoint ... --forecast-date YYYY-MM-DD --output 新目录`。使用目标日之前10天已存在的日帧，不使用目标日标签。数据／配置必须与训练指纹一致，本版本适用于固定目录内指定日期或目录末尾次日；不伪装支持追加新目录的生产部署。目录末日若未确认完整监测，会拒绝拿它预测次日。应先解决数据完整性，不通过删除标记绕过检查。

## 9. 自检与已验证范围

```powershell
python -m unittest discover -s risk_6306 -p test_core.py -v
```

测试覆盖能量／计数守恒、边界事件落格、硬标签、面积并列处理、归一化不使用目标与未来、零报警阈值，以及安装PyTorch时的模型前向／反向检查。

交付环境已使用本次上传的原始 `6306_all.csv`，通过仓库现有预处理脚本重建源数据，运行完整新数据准备与验证期基线。训练676样本、40次高能事件；验证365样本、6次；测试349样本、9次。测试只统计样本数量，未运行测试评价。本目录 `validation_reference` 提供可复核的真实统计与基线结果。

交付环境没有PyTorch，因此模型训练、GPU运行和模型前向／反向测试尚未执行；对应单元测试自动跳过。所有Python脚本均通过语法编译检查，其余四项单元测试通过。训练效果需要在env2实际运行后判断。

先运行准备、基线和 `all_seed42` 三步，再检查结果。最适合发回分析的是 `split_summary.json`、`training_summary.json`、`loss_log.csv`、`val/summary.json` 和 `val/frozen_threshold_metrics.json`。
