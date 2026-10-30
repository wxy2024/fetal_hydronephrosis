# RCC-Pooling图像质量扰动实验

## 运行
将scripts/evaluate_rcc_perturbation.py同步到服务器main8的scripts文件夹。
使用能运行原FetalCLIP训练脚本的环境。在服务器项目目录执行：

    python scripts\evaluate_rcc_perturbation.py

所有路径和参数在脚本顶部，已填写你提供的服务器路径；不会读取当前配置里的模型类型或修改训练配置。
先设置MAX_PATIENTS = 2试运行并检查examples图像；确认后设置0评估全部符合条件患者。每次新建结果目录，不覆盖旧结果。
依赖原训练环境，以及numpy、pandas、Pillow、matplotlib。

## 必须具备
- 两个结果目录各自的5fold_predictions.csv，包含image_attention_json。
- 每种模型的五折最终分类模型权重（不是只有FetalCLIP_weights.pt）。
- 原来的fetalclip/fetalclip_wrapper.py、FetalCLIP配置和预训练权重。
- 原始数据及报告。

重要：本地现有训练脚本没有保存分类模型权重的语句，因此需确认服务器确实保存了权重！若只有CSV则无法执行本实验。脚本不会重新训练，也不会用随机模型替代。
自动搜索文件路径中包含fold_1等名称的.pt/.pth/.ckpt；每折必须唯一。如有多个候选，在CHECKPOINTS中明确填写最终外层训练模型，不能填内层验证模型。例如：

    CHECKPOINTS['rcc_pooling'] = {
        'fold_1': Path(r'E:\实际目录\fold_1\final_model.pt'),
        # 继续填fold_2至fold_5
    }

仅加载安全的tensor state_dict，支持model_state_dict/state_dict/model字典包装及module.前缀。
两模型必须同一队列、同一测试折、标签及完整检查历史；不匹配时停止，不擅自取交集或重建折。
模型结构参数在MODEL_KWARGS。加载严格核对参数形状，且对每位选中患者重算原始概率和权重，与保存结果比较（容差1e-4）；不一致则停止，不能通过放大容差掩盖配置问题。

## 实验设置
固定种子42；每位患者随机选一个含至少两张图的肾单位检查，轮流扰动该检查每张图，其余图像、报告、结构化信息及完整随访历史不变。抽样不依赖权重或标签。
- 模糊：强度1/2/4；在原图上按短边/224换算GaussianBlur半径，保留原几何尺寸。
- 噪声：[0,1]像素上添加标准差0.02/0.05/0.10的高斯噪声，截断至[0,1]；重复5次，同一重复在不同强度使用同一噪声方向。
处理后保存临时PNG，再调用原extractor.encode_images，因此仍经过原模型预处理。只替换目标图的特征，不覆盖原图。模糊采用Pillow近似高斯滤波，不是固定预处理后张量的精确sigma。
两个模型共享同一份扰动图像及特征；推理为eval模式，模型不更新。示例只用于检查人为退化强度，不能按模型表现筛选扰动设置。

## 输出
保存到output6_rebuttal/rcc_perturbation/时间戳：
- selection.csv：患者与检查的固定抽样清单。
- manifest.json：参数、模型/预测文件哈希及运行状态；只有status=complete才是完成结果。
- baseline_replay.csv：原始预测复现核对。
- perturbations.csv：逐图、逐强度、逐重复的权重/排名/概率变化。
- patient_effects.csv：先平均重复，再平均图像，得到患者级效果。
- summary.csv：患者平均效果及10,000次患者Bootstrap百分位95%区间。
- paired_differences.csv：同一患者RCC减标准注意力的效果差及配对Bootstrap区间。
- patient_trends.csv：各患者图像权重非递增/严格递减比例（先对噪声重复取均值）。
- weight_change.png：两模型权重变化曲线与区间。
- examples：第一位选中患者的第一张原图及扰动例图。

weight_change负值表示降权；rank_change正值表示排名下降（1为最高，平局取平均）；abs_probability_change越小表示预测改变越小。weight_decreased以1e-8为数值容差。
主要终点可预先定为重度模糊/噪声的权重变化，其他结果为探索性。报告同次检查的相对权重，不能将较大降权直接理解成可靠性更高：原始权重分布不同也会影响变化范围。本脚本报告区间，不输出多指标显著性P值；单患者试跑不计算区间。
不将图像和噪声重复视为独立患者。与医生评分结果共同解释；人工扰动响应不能证明临床可靠性或因果机制。

## 故障排查
缺权重：检查服务器是否有五折最终模型；只有预测CSV无法恢复模型。
折/队列不一致：需先统一实验协议，脚本不会继续不公平对比。
原始结果无法复现：检查FetalCLIP权重、模型配置、数据内容及预处理版本。
中途失败会保留已有CSV便于诊断，但该目录不是完整实验结果。修复后重新运行创建新目录。
本地仅验证了计算和接口逻辑；真实模型加载及原预测复现需要服务器上的权重和wrapper验证。
