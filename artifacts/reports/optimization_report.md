# D-FINE 手术器械检测持续优化报告

生成日期：2026-08-09

## 结论

最终最佳版本为 **D-FINE-M 640，第 20 epoch**。模型用 VID001_B 训练、VID001_C 验证选型、VID001_D 固定测试；D 从未参与模型、阈值或时序参数选择。最佳权重为：

`/root/autodl-tmp/surgical/artifacts/experiments/dfine_m_640/best_stg1.pth`

相对 RT-DETRv2-R50 基线，D 集官方 COCO mAP 从 0.094 提高到 0.160（约 +70%），mAP50 从 0.312 提高到 0.509（约 +63%），mAP75 从 0.033 提高到 0.046（约 +38%）。验证集 C mAP 从 0.273 提高到 0.320（约 +17%）。

## 技术选择

D-FINE 是 ICLR 2025 Spotlight 的实时 DETR，使用细粒度分布精修和全局最优定位自蒸馏加强边界定位，且自蒸馏不增加推理成本。本项目使用官方代码 commit `267a6da6d04c8ad52e54120692896515b9e55981` 和 COCO 预训练 D-FINE-M 权重。[D-FINE 论文](https://arxiv.org/abs/2410.13842)，[官方代码](https://github.com/Peterande/D-FINE)

候选排序使用两类后处理消融：标签无关的 Viterbi 全序列重排序用于完整视频；另用连续时间 5 折 OOF 的梯度提升候选 IoU 校准估计标注正帧上界。后者因缺少无器械负帧而不用于完整视频。

## 完整实验比较

| 方案 | 完整训练 | C mAP | C mAP50 | C mAP75 | 结论 |
|---|---:|---:|---:|---:|---|
| RT-DETRv2-R50 640 | 19 epoch（早停） | 0.273 | 0.610 | 0.197 | 原基线 |
| D-FINE-M 640 | 30 epoch | **0.320** | **0.718** | **0.239** | 最佳，第 20 epoch |
| D-FINE-M 800 | 14 epoch | 0.280 | 0.658 | 0.204 | 升分辨率无增益 |
| D-FINE-M 640 完整框低 LR 精修 | 8 epoch | 0.288 | 0.651 | 0.212 | 继续拟合 B，域泛化下降 |

D-FINE-M 640 的固定最佳权重官方复测：

| 数据段 | 用途 | mAP | mAP50 | mAP75 | AR100 |
|---|---|---:|---:|---:|---:|
| B | train | 0.824 | 1.000 | 0.979 | 0.893 |
| C | val | 0.320 | 0.718 | 0.239 | 0.601 |
| D | test | 0.160 | 0.509 | 0.046 | 0.552 |

## 候选排序消融

| 选择方法 | C top1 F1@IoU50 | C mean IoU | D top1 F1@IoU50 | D mean IoU |
|---|---:|---:|---:|---:|
| D-FINE framewise top1 | 0.676 | 0.545 | 0.549 | 0.472 |
| Viterbi（C 调参） | 0.706 | 0.575 | 0.594 | 0.517 |
| 学习式校准（C 5-fold OOF） | 0.718 | 0.606 | **0.603** | **0.528** |

学习式校准在全为阳性的标注帧上选择阈值 0.01，说明当前数据没有提供足够负帧来学习“器械不存在”状态。因此完整视频采用 Viterbi 的 C-selected 阈值 0.43。

## 三段完整视频

| 视频 | 帧数 | 模型 FPS（不含 IO） | 新检测率 | 旧检测率 | 单一轨迹覆盖 | 运动学可靠 |
|---|---:|---:|---:|---:|---:|---|
| VID001_B | 3000 | 159.1 | 84.1% | 77.1% | 84.1% | 是 |
| VID001_C | 3000 | 158.3 | 76.6% | 63.0% | 76.6% | 否 |
| VID001_D | 3001 | 157.7 | 59.5% | 36.4% | 59.5% | 否 |

新时序路径统一使用单一 track ID，消除了旧 IoU tracker 的身份碎裂。按预设覆盖率 80% 门槛，B 的轨迹运动学可用；C 和 D 虽明显改善，仍不能据此给出稳定扶镜质量结论。

## Debug：剩余误差来源

D 的前 20 候选召回率为 92.0%，说明主问题不是“模型完全看不到器械”，而是边界定位和候选排序。Viterbi 后 515 帧的互斥错误计数为：

- 208 帧：IoU 0.50–0.75，边界仍不精确；
- 130 帧：存在合格候选但时序排序选错；
- 76 帧：IoU ≥ 0.75；
- 74 帧：分数低于 C-selected 0.43；
- 27 帧：前 20 候选均无 IoU ≥ 0.5 的框。

分组诊断显示 D 的 medium 框 IoU50 命中率 79.5%，small 为 61.0%，large 仅 37.2%；边缘位置为 41.2%，中心位置为 59.8%。大框最差与人工标注范围不一致相符：部分框覆盖完整器械杆，模型更常稳定定位工作端。C 的后半段也显著弱于前半段，表明同一视频内部仍存在成像/阶段域漂移。

## 下一步优化优先级

1. **先统一标签协议并复核大框**：明确标注“工作头”还是“工作头+器械杆”，双人复核 D 大框和边缘框。该项比继续盲目调分辨率更可能提高 mAP75。
2. **补充负帧与困难负样本**：从无目标阶段抽取并人工确认无器械帧，才能训练可靠的 missing state、学习式重排序和全视频阈值。
3. **增加独立病例**：当前 B/C/D 均来自 VID001，必须按病例拆分至少 3 个病例；现结果只能称 single-case segment holdout。
4. **训练监督升级**：可尝试 RT-DETRv3 的层级密集正样本监督，或 DEIM 的改进匹配，以缓解 DETR 一对一匹配在小数据上的稀疏监督与收敛问题。[RT-DETRv3](https://arxiv.org/abs/2409.08475)，[DEIM（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.pdf)
5. **用真正的视频模型替代手工时序项**：加入 6–8 Hz 左右的短时上下文、可学习 temporal queries/memory，并显式建模无目标状态。近期手术视频 Transformer 工作也报告了细粒度时间上下文的收益。[Surgical instrument-tissue interaction video transformer](https://pubmed.ncbi.nlm.nih.gov/41214416/)

## 可复现资产

- 最佳模型指针：`artifacts/experiments/best_model.json`
- 全部实验注册：`artifacts/experiments/registry.json`
- 640 训练日志与权重：`artifacts/experiments/dfine_m_640/`
- Viterbi 消融：`artifacts/experiments/dfine_m_640/temporal/summary.json`
- 学习式重排序：`artifacts/experiments/dfine_m_640/learned_rerank/summary.json`
- 三段完整推理：`artifacts/inference_best/VID001_{B,C,D}/`
- 扶镜指标：`artifacts/metrics_best/camera_metrics.csv`
- 误差摘要：`artifacts/error_analysis/dfine_temporal_summary.json`

## 有效性边界

这是完整训练、完整标注帧评估和三段完整视频推理，不是 smoke test。D 视频当前可完整解码并已用于固定测试。但三段属于同一病例、完整视频无逐帧负标签、专家 SALAS/OSA-CNS 评分尚未填入，因此暂时不能完成病例级临床效度、ICC、相关性或专家一致性结论。
