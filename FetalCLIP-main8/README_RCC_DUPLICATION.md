# RCC 与标准注意力：重复图像实验

## 运行

无需复制整个项目、修改训练流程或重新训练。将新增的 `scripts/evaluate_rcc_duplication.py` 同步到服务器同名目录，在原来的 `wxy_fetalclip` 环境运行：

```powershell
conda activate wxy_fetalclip
cd E:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main8
python scripts\evaluate_rcc_duplication.py
```

首次默认 `SMOKE_TEST = True`，运行固定队列中的2例。成功后将脚本顶部改为 `SMOKE_TEST = False`，再次执行同一命令，完成185例正式实验。所有路径在脚本顶部修改，不需要交互输入。

默认路径已配置为服务器地址：

- 数据：`E:\wxy_codes\dataset_hydro_1234\raw`
- 模型：main8 下 `output6_rebuttal` 中的 `rebuttal_69_dataset_1234_...` 和 `rebuttal_1013_dataset_1234_...`，每折需有 `final_model.pt`。
- FetalCLIP：main8 根目录的 `FetalCLIP_weights.pt`、`FetalCLIP_config.json`，以及服务器原有的 `fetalclip` 包。
- 固定患者清单：`output6_rebuttal\rcc_perturbation\20261031_183355_665139\selection.csv` 和同目录 `manifest.json`。请保留它们。

## 实验做什么

沿用上次正式实验的185例，每例同一个肾单位、同一次检查；选中的检查各有2–10张图片，合计619张。其余检查允许没有图像。两个模型使用各自对应的五折测试患者权重。

对这619张图逐张操作：在原检查中额外加入该图的1、3、5个完全相同副本，每次从原始输入开始。保留其他图片、报告、检查时间及完整随访历史。正式实验共 `619 × 3 × 2 = 3714` 条扰动结果。

FetalCLIP冻结且推理确定，因此直接复制其已提取特征，等价于加入同图的副本，可省去重复编码；不会写入或改变原始图片。该实验针对完全重复保存，不涵盖不同图像的近似视角。

先核对原始预测和全部注意力能否复现，并校验患者分折、图片顺序、预测文件及模型权重哈希；不一致会中止，不会自动删患者或换权重。

## 看哪些结果

自动保存到 `output6_rebuttal\rcc_duplication\SMOKE_时间戳` 或 `FORMAL_时间戳`。正式分析只用 `FORMAL_` 且 `manifest.json` 中 `status` 为 `complete` 的结果。

- `summary.csv`：两模型各重复次数的指标均值及95%置信区间。
- `paired_differences.csv`：RCC减标准注意力的配对差值及95%置信区间，优先看此文件。
- `duplication_effects.png`：权重、表示和预测变化曲线。
- `duplication_results.csv`：每张原图、每种重复次数的详细结果。
- `patient_effects.csv`：先在每位患者内部平均后的结果。
- `baseline_weights.csv`：原始注意力α及实际聚合系数β。
- `mixture_coefficients.csv`：每折学到的混合系数λ。
- `baseline_replay.csv`、`selection.csv`、`manifest.json`：复现、患者清单和参数记录。

指标含义：

- `group_weight_change`：原图及新增副本的**总聚合系数**减去原图原来的系数，衡量重复是否导致该图占比增加。不是只看单张副本的权重。原检查中其他相似图像不自动归为同组。
- `relative_feature_change`：聚合表示变化的L2范数除以原表示L2范数。
- `cosine_distance`：聚合表示变化前后的余弦距离。
- `abs_probability_change`：最终阳性概率变化的绝对值。

RCC输出混合了均值分支和注意力分支，所以使用实际系数 `β=(1−λ)/N+λ×α`；标准注意力使用 `β=α`。脚本验证β能够重建模型实际使用的聚合表示。

预先指定主要比较为：**额外加入5个副本时，relative_feature_change 的 RCC−attention 差值**。负值表示RCC变化较小；95%区间全小于0支持其在这一指标上更稳定。其他指标/重复次数作为补充分析，避免只挑最有利结果。

先对每位患者的目标图片取平均，再在原测试折内按患者配对Bootstrap 10,000次，保证患者等权、两个模型抽到同批患者。置信区间以这批已训练模型为条件，不包含重新训练波动；次要比较区间未做多重校正，本脚本不计算P值。冒烟测试仅用于确认流程，不解读其置信区间。

变化较小支持对重复输入更稳定，不直接证明准确率更高或权重代表临床图像质量；若差值区间跨0，应如实报告证据尚不充分。
