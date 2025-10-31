import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)
        
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
        # =========================================================================
        # 这部分代码完全按照你的要求保持不变
        # =========================================================================
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)

        else:
            # =========================================================================
            # START: 核心修改区域
            # 我们将用下面的智能逻辑替换原来简单的 `outputs = self.model(**inputs)`
            # =========================================================================

            # Step 1: 统一解包模型，找到真正的基础模型类型
            # 这个逻辑对所有模型都运行，以确保我们总是基于基础架构进行判断。
            print("[DEBUG] ===== Start unwrapping model =====")
            model_ref = self.model
            unwrap_depth = 0
            while hasattr(model_ref, "base_model") or hasattr(model_ref, "model"):
                unwrap_depth += 1
                cls_name = model_ref.__class__.__name__
                print(f"[DEBUG] Unwrapped level {unwrap_depth}: {cls_name}")
                if hasattr(model_ref, "base_model"):
                    model_ref = model_ref.base_model
                elif hasattr(model_ref, "model"):
                    model_ref = model_ref.model
                else:
                    break
                if unwrap_depth > 50:
                    print(f"[DEBUG] WARNING: unwrap depth > 50, may indicate recursive model structure!")
                    break
            
            # 解包循环结束后，model_ref 是最内层的模型。获取其类名。
            base_model_name = model_ref.__class__.__name__.lower()
            print(f"[DEBUG] Final resolved model class: {base_model_name}")
            print("[DEBUG] ===== End unwrapping model =====")

            # Step 2: 根据解析出的基础模型类名，准备输入并调用模型
            if any(k in base_model_name for k in ["esmc", "evolutionaryscale"]):
                # --- ESMC 模型特殊处理路径 ---
                print("[DEBUG] ESMC forward path selected.")
                
                # 创建一个干净的参数字典，以避免向底层模型传递冲突或不支持的参数。
                esmc_kwargs = inputs.copy()

                # ---- STEP 2.1: 向后兼容 'sequences' 键 (来自你的原始代码)
                seq_obj = esmc_kwargs.get("sequences", None)
                if seq_obj is not None and "input_ids" not in esmc_kwargs and "sequence_tokens" not in esmc_kwargs:
                    if isinstance(seq_obj, (list, tuple)) and len(seq_obj) > 0 and isinstance(seq_obj[0], str):
                        tokens = self.model.tokenizer(seq_obj, return_tensors="pt", padding=True, truncation=True)
                        device_ = next(self.model.parameters()).device
                        esmc_kwargs["input_ids"] = tokens["input_ids"].to(device_)
                        esmc_kwargs["attention_mask"] = tokens["attention_mask"].to(device_)
                    elif isinstance(seq_obj, torch.Tensor):
                        esmc_kwargs["sequence_tokens"] = seq_obj
                    else:
                        raise TypeError(f"[SaProtClassificationModel] Unexpected type under 'sequences': {type(seq_obj)}")

                # ---- STEP 2.2: 统一输入字段名，这是修复 TypeError 的关键
                # ESMC 模型期望 'sequence_tokens' 而不是 'input_ids'。
                # 我们使用 .pop() 来获取值并移除旧键，从而避免参数冲突。
                if "input_ids" in esmc_kwargs:
                    print("[DEBUG] Renaming 'input_ids' to 'sequence_tokens' for ESMC.")
                    esmc_kwargs["sequence_tokens"] = esmc_kwargs.pop("input_ids")
                elif "sequence_tokens" not in esmc_kwargs:
                    raise ValueError(
                        "[SaProtClassificationModel] ESMC forward requires tokenized tensors "
                        "under 'input_ids' or 'sequence_tokens'."
                    )
                
                # ---- STEP 2.3: 移除 ESMC 不支持的参数
                # 基础 ESMC 模型的 forward 方法不接受 'attention_mask'。
                # PEFT/LoRA 包装器知道如何处理这种情况，但我们不应将此参数传递给最底层。
                if "attention_mask" in esmc_kwargs:
                    print("[DEBUG] Removing 'attention_mask' for ESMC call.")
                    esmc_kwargs.pop("attention_mask")

                # 使用为 ESMC精心准备的、干净的参数字典来调用模型。
                outputs = self.model(**esmc_kwargs)

            else:
                # --- 标准 Hugging Face 模型路径 (例如 EsmForSequenceClassification, ProtBertForSequenceClassification 等) ---
                print("[DEBUG] Standard Hugging Face model forward path selected.")
                
                # 这些模型期望标准的 'input_ids', 'attention_mask' 等。
                # 原始的 'inputs' 字典已经是正确的格式。
                outputs = self.model(**inputs)

            # =========================================================================
            # END: 核心修改区域
            # =========================================================================

            # =========================================================================
            # 这部分代码完全按照你的要求保持不变，它现在处理上面逻辑块生成的 `outputs`
            # =========================================================================
            if isinstance(outputs, dict):
                logits = outputs.get("logits", list(outputs.values())[0])
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
                
        return logits

    def loss_func(self, stage, logits, labels):
        if not isinstance(logits, torch.Tensor):
            if hasattr(logits, "logits"):
                logits = logits.logits
            elif hasattr(logits, "sequence_logits"):
                logits = logits.sequence_logits
            elif isinstance(logits, dict):
                if "logits" in logits:
                    logits = logits["logits"]
                elif "sequence_logits" in logits:
                    logits = logits["sequence_logits"]
                else:
                    raise TypeError(f"[SaProtClassificationModel] logits dict has no proper key: {logits.keys()}")
            elif hasattr(logits, "__getitem__"):
                logits = logits[0]
            else:
                raise TypeError(f"[SaProtClassificationModel] logits must be Tensor, got {type(logits)}")
        label = labels["labels"]

        # Automatic aggregation for token-level outputs such as ESMC
        if logits.ndim == 3:
            logits = logits.mean(dim=1)
        if label.ndim > 1:
            label = label.squeeze(-1)

        loss = cross_entropy(logits, label)

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metric
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        # log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)