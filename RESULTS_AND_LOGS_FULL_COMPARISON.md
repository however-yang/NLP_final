# Baseline 与微调结果对比（单次完整训练链路）

## 微调设置（摘要）

- **基座模型**：`Qwen/Qwen3-VL-8B-Instruct`（`configs/model/backbone_main.yaml`，bfloat16）。
- **参数高效微调**：LoRA / PEFT（`configs/model/peft.yaml`：`r=48`，`lora_alpha=96`，`lora_dropout=0.05`，`target_modules` 为 `q_proj/k_proj/v_proj/o_proj`，`bias=none`，`task_type=CAUSAL_LM`）。
- **任务与数据**：DocVQA + ChartQA 联合训练（`configs/train/train_joint.yaml`：`prompt_style: structured`，平衡采样；训练/验证均为已构建的 `data/processed/.../jsonl`）。
- **训练技巧（与当前 yaml 一致）**：`gradient_checkpointing: true`，`batch_size=2`，`gradient_accumulation_steps=8`，`image_max_pixels=1003520`，`bf16: true`，`max_answer_tokens=16`；按 step 保存与验证（`save_steps` / `eval_steps: 100`），`num_train_epochs: 2`。
- **本次对比所指的「完整一次训练」**：日志与清单上为从 **`checkpoint-1000` 续训至结束** 的流水线（`run_20260421_105335`），磁盘上最终用于评测的适配器为 **`checkpoint-1250`**。验证协议与 Baseline 一致：**同一验证 jsonl + structured 推理**。

---

## 结果（Baseline vs 微调）

数据来源：**Baseline** 取 `outputs/___BASELINE_FINAL_RESULTS___/run_20260419_225040/metrics/baseline_structured_{docvqa,chartqa}_val__20260419_225040.json`；**微调** 取 `outputs/___CHECKPOINT_VALIDATION_RESULTS___/run_20260421_150432/report_checkpoint-1250.json`（与阶段 6 选优一致）。

| 指标 | Baseline（structured） | 微调（checkpoint-1250） |
| --- | ---: | ---: |
| DocVQA（val，n=1250） | 0.138 | **0.761** |
| ChartQA（val，n=1920） | 0.075 | **0.681** |
| 合并验证集 overall（n=3170） | — | **0.712** |

**结论**：在相同验证数据与 **structured** 设定下，相对 Baseline，DocVQA 与 ChartQA 均有 **大幅度** 提升；合并验证集上整体约 **71%** 平均得分量级（见报告中的 `overall`）。
