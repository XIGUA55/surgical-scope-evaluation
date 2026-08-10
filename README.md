# 胰腺手术扶镜质量评价复现

本仓库复现 PPT 中的完整技术路线：在腹腔镜胰腺手术关键阶段检测并追踪主操作器械头，计算器械相对视野中心的位置与运动学指标，再与专家 SALAS / OSA-CNS 评分进行一致性、相关性和病例级验证。

检测器采用 Transformer 架构，不使用 YOLO。初始基线为 **RT-DETRv2-R50**，当前优化主线为 **D-FINE-M**（ICLR 2025 Spotlight），并使用验证集调参的 Viterbi 时序候选重排序。标签格式为：

```text
class_id,object_id,cx,cy,width,height
```

坐标均为归一化值。当前类别 `0` 对应 `Harmonic0`，第二列是对象 ID，不是检测类别。

> 数据说明：原始手术视频、逐帧图像、标注压缩包、模型权重和包含手术画面的可视化结果不提交到 Git 仓库。复现实验前，请将获授权的数据放入本地 `data/` 目录；仓库仅保留代码、配置、测试和不含图像的轻量实验摘要。

## 当前数据状态

- `VID001_B_5fps.mp4`、`VID001_C_5fps.mp4`、`VID001_D_5fps.mp4` 及三组标签均已通过审计，可完整解码；
- 当前分段留出为 B（1456 帧）训练、C（1882 帧）验证、D（515 帧）测试；模型和阈值只按 C 选择，D 不参与调参；
- B/C/D 都属于同一个病例 `VID001`。该完整实验适合复现和段间域偏移分析，但不能替代至少 3 个独立病例的病例级泛化验证。

## 环境

```bash
conda activate surgical-repro
cd /root/autodl-tmp/surgical
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e '.[dev]'
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e '.[dfine]'
```

PyTorch 固定为 2.8.0、torchvision 固定为 0.23.0，对应 CUDA 12.8 构建。可用以下命令确认：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 完整分段留出实验

```bash
scope-eval split --segment-holdout
scope-eval train
scope-eval evaluate --checkpoint artifacts/runs/formal/best --split test
scope-eval analyze-errors --checkpoint artifacts/runs/formal/best
```

上述 RT-DETRv2 基线已完成。优化实验位于 `config/dfine_m_surgical.yml`（原始 VID001_B 训练）、`config/dfine_m_surgical_multicase.yml`（VID001_B + CH2001_CH001 多病例训练）和 `config/dfine_m_surgical_800.yml`（小目标高分辨率精修）。所有版本由 `artifacts/experiments/registry.json` 按 C 集 mAP 统一登记，当前最佳指针写入 `artifacts/experiments/best_model.json`。数据准备只纳入存在标签文件的帧；缺失标签的帧视为筛除帧，不作为负样本。

标注质检候选可通过 `scope-eval sample-anomalies --samples-per-reason 4` 生成。输出位于 `artifacts/audit/anomaly_samples`，同时保存原图、框图、contact sheet 和完整 CSV 索引；候选不会被自动剔除。

D-FINE 完成后，可提取候选并只在 C 上选择时序参数，再对 D 作固定参数测试：

```bash
scope-eval dfine-candidates --dfine-config config/dfine_m_surgical.yml \
  --checkpoint artifacts/experiments/dfine_m_640/best_stg1.pth --split val \
  --output artifacts/experiments/dfine_m_640/candidates_val.csv
scope-eval dfine-candidates --dfine-config config/dfine_m_surgical.yml \
  --checkpoint artifacts/experiments/dfine_m_640/best_stg1.pth --split test \
  --output artifacts/experiments/dfine_m_640/candidates_test.csv
scope-eval temporal-evaluate \
  --validation-candidates artifacts/experiments/dfine_m_640/candidates_val.csv \
  --test-candidates artifacts/experiments/dfine_m_640/candidates_test.csv \
  --output-dir artifacts/experiments/dfine_m_640/temporal
```

多个视频可在一次指标计算中传入：`scope-eval metrics --tracks B_tracks.csv C_tracks.csv D_tracks.csv`。

正式多病例数据到齐后运行病例级 `scope-eval split` 和训练；少于 3 个独立病例时该划分会明确终止。最终建议报告 mAP50、mAP50-95、precision/recall、检测率、居中率、中心距离、速度、加速度、jerk、抖动、重居中延迟、专家 ICC(2,1)、Spearman 相关和 leave-one-case-out 预测结果。

当前方案不会发生相邻帧随机拆分泄漏，但三段仍来自同一病例，所以结果必须标注为 single-case segment holdout。

主要输出位于 `artifacts/`：

- `audit/dataset_audit.json`：视频完整性和标签审计；
- `dataset/`：抽帧图像、COCO 标注、manifest 与划分；
- `runs/{smoke,formal}/`：权重、处理器、训练历史；
- `evaluation/`、`inference/`、`metrics/`、`ratings/`：各阶段结果；
- `reports/report.{md,html}`：汇总报告。

专家评分模板位于 `ratings/expert_scores_template.csv`，必需列为 `case_id, segment_id, rater_id, salas_total, osa_cns_total`。
