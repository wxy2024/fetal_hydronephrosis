# 五折最终模型保存

本次新增保存，不改变五折划分、内层选轮次或外层训练流程。
同步到服务器的文件：
- scripts/v1_train_5fold_online_attn_gru.py
- utils/checkpoint_io.py

训练前在configs/config_online_attn.py设置新的EXP_ID，避免覆盖原结果。
当前配置是什么模型，就保存什么模型；本次没有自动切换为完整模型或更改随机种子。
如重训完整模型，确认IMAGE_POOLING_TYPE='rcc_pooling'、TEMPORAL_BACKBONE='td_conv_gru'、INPUT_MODALITY='multimodal'、TEMPORAL_EXPERIMENT='baseline'。
本次没有新增固定实验69划分功能；如要复用其划分，需核对训练所用种子、数据和最终患者划分，不能仅因是五折就认为相同。

原命令不变：

    python scripts\v1_train_5fold_online_attn_gru.py

每折完成内层轮次选择、在整个外层训练集重训后，立即保存：

    实验结果目录/fold_1/final_model.pt
    实验结果目录/fold_1/final_model.json
    实验结果目录/fold_1/predictions.csv

fold_2至fold_5同理。权重在外层测试前保存，predictions.csv在该折测试完成后保存。五折结束仍生成原有汇总CSV。
.pt包含model_state_dict和metadata，JSON为同一元数据的可读副本：模型构造参数、种子、轮次、训练/测试患者与样本ID、训练配置、FetalCLIP路径、内层轮次选择记录。
这是最终推理权重，不含优化器状态，不能用于断点续训；FetalCLIP预训练权重仍需另行保留。
若结果目录非空，启动即报错并保留原文件。中断后已完成折仍在；重新运行需使用新EXP_ID，不自动跳过已完成折。

evaluate_rcc_perturbation.py能够自动识别此目录结构。使用时指向本次新结果目录（两个模型各自的新目录），不要将新权重与旧预测CSV混用。
本地通过权重往返加载、预测一致性、训练测试患者重叠检查及防覆盖测试；未启动真实训练。
