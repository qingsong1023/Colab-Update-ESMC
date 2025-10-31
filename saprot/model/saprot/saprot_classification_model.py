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
        # ==============================================================
        # isolate ESMC model (handle LoRA / PEFT wrapping)
        # ==============================================================
        try:
            print("\n================ DEBUG: forward() called ==================")
            print(f"Initial model class: {self.model.__class__.__name__}")

            # -------------------- unwrap LoRA / PEFT  --------------------
            model_ref = self.model
            unwrap_depth = 0
            while hasattr(model_ref, "base_model") or hasattr(model_ref, "model"):
                unwrap_depth += 1
                if hasattr(model_ref, "base_model"):
                    model_ref = model_ref.base_model
                elif hasattr(model_ref, "model"):
                    model_ref = model_ref.model
                else:
                    break
                print(f"[DEBUG] Unwrapped level {unwrap_depth}: {model_ref.__class__.__name__}")

            real_cls_name = model_ref.__class__.__name__.lower()
            print(f"[DEBUG] Final detected inner model class: {real_cls_name}")
            print(f"[DEBUG] Input keys before mapping: {list(inputs.keys())}")

            if "esmc" in real_cls_name or "evolutionaryscale" in real_cls_name:
                print("[DEBUG] Detected ESMC model expecting tokenized Tensor input.")

                # ---- STEP 1: 兼容旧字段 sequences (可能是 list[str] / Tensor)
                if "sequences" in inputs and "input_ids" not in inputs and "sequence_tokens" not in inputs:
                    seq_obj = inputs["sequences"]

                    # 判断类型: 如果是 list[str]，自动调用 tokenizer
                    if isinstance(seq_obj, (list, tuple)) and len(seq_obj) > 0 and isinstance(seq_obj[0], str):
                        print("[DEBUG] 'sequences' detected as list of str → Auto-tokenizing with model.tokenizer() ...")
                        tokens = self.model.tokenizer(
                            seq_obj, return_tensors='pt', padding=True, truncation=True
                        )
                        device_ = next(self.model.parameters()).device
                        inputs["input_ids"] = tokens["input_ids"].to(device_)
                        inputs["attention_mask"] = tokens["attention_mask"].to(device_)
                        print(f"[DEBUG] Tokenization complete → 'input_ids' shape: {inputs['input_ids'].shape}")

                    # 否则可能已经是 tensor（旧式 collate_fn 输出）
                    elif isinstance(seq_obj, torch.Tensor):
                        print("[DEBUG] 'sequences' detected as Tensor mapping to 'sequence_tokens'")
                        inputs["sequence_tokens"] = seq_obj
                    else:
                        raise TypeError(
                            f"[SaProtClassificationModel] Unexpected data type under 'sequences': {type(seq_obj)}"
                        )

                # ---- STEP 2: 检查必须有 tokenized tensor
                if "input_ids" not in inputs and "sequence_tokens" not in inputs:
                    raise ValueError(
                        "[SaProtClassificationModel] ESMC forward expects tokenized tensor under 'input_ids' "
                        "(please call esm_model.tokenizer() or alphabet.batch_converter() before forward)."
                    )

                # ---- STEP3: 向后兼容 'sequence_tokens'
                if "sequence_tokens" not in inputs and "input_ids" in inputs:
                    inputs["sequence_tokens"] = inputs["input_ids"]

                # ==============================================================
                # 真正地 forward 调用模型
                # ==============================================================
                outputs = self.model(**inputs)
                print("[DEBUG] Forwarded through ESMC successfully")

                # ---- STEP 4: 返回统一输出
                if isinstance(outputs, dict):
                    print(f"[DEBUG] Output keys: {list(outputs.keys())}")
                    return outputs.get("logits", list(outputs.values())[0])
                return outputs

            # ==============================================================
            # 非 ESMC 模型：保持原有逻辑，不修改
            # ==============================================================
            else:
                print("[DEBUG] Not an ESMC model, using default logic.")

        except Exception as e:
            print(f"[SaProtClassificationModel] ESMC forward isolation skipped: {e}")

        # ==============================================================
        # end isolate ESMC model
        # ==============================================================

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
            logits = self.model(**inputs).logits
        
        return logits

    def loss_func(self, stage, logits, labels):
        # ======== Debug: print ESMC output structure ========
        if not isinstance(logits, torch.Tensor):
            print("\n[DEBUG] loss_func called with non-Tensor logits")
            print("[DEBUG] type:", type(logits))
            try:
                # 如果是一个 dataclass 或 namedtuple，有 __dict__
                if hasattr(logits, "__dict__"):
                    print("[DEBUG] __dict__ keys:", list(logits.__dict__.keys()))
                # 如果是普通命名元组，打印 _fields
                elif hasattr(logits, "_fields"):
                    print("[DEBUG] _fields:", logits._fields)
                # 如果是 dict
                elif isinstance(logits, dict):
                    print("[DEBUG] dict keys:", list(logits.keys()))
                else:
                    # 最后兜底打印所有可访问属性
                    print("[DEBUG] dir():", [k for k in dir(logits) if not k.startswith("_")])
            except Exception as e:
                print("[DEBUG] Failed to inspect logits:", e)
        # ======== Debug end ========

        # 暂时先不要 raise，让它真正报错前能打印出结构
        if not isinstance(logits, torch.Tensor):
            if hasattr(logits, "logits"):
                logits = logits.logits
            elif isinstance(logits, dict) and "logits" in logits:
                logits = logits["logits"]
            else:
                raise TypeError(f"[SaProtClassificationModel] logits must be Tensor, got {type(logits)}")

        label = labels['labels']
        loss = cross_entropy(logits, label)

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
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