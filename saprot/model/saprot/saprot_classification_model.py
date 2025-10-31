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
        # add esmc start
        # ==============================================================
        model_cls_name = self.model.__class__.__name__.lower()
        # Step 0 - Resolve true model class (debug unwrap)
        if any(k in model_cls_name for k in ["esmc", "evolutionaryscale"]):
            # ==============================================================  
            # [Special Handling for ESMC Models]
            # ==============================================================  
            print("[DEBUG] Detected ESMC model, entering special forward path.")
            model_ref = self.model
            unwrap_depth = 0
            while hasattr(model_ref, "base_model") or hasattr(model_ref, "model"):
                unwrap_depth += 1
                print(f"[DEBUG] Unwrapping level {unwrap_depth}: {model_ref.__class__.__name__}")
                if hasattr(model_ref, "base_model"):
                    model_ref = model_ref.base_model
                elif hasattr(model_ref, "model"):
                    model_ref = model_ref.model
                else:
                    break
            real_cls_name = model_ref.__class__.__name__.lower()
            print(f"[DEBUG] Final resolved class: {real_cls_name}")

            # Step 1 - ESMC / EvolutionaryScale Models
            seq_obj = inputs.get("sequences", None)
            # ---- STEP 1.1: backward compatibility for 'sequences'
            if seq_obj is not None and "input_ids" not in inputs and "sequence_tokens" not in inputs:
                if isinstance(seq_obj, (list, tuple)) and len(seq_obj) > 0 and isinstance(seq_obj[0], str):
                    tokens = self.model.tokenizer(seq_obj, return_tensors="pt", padding=True, truncation=True)
                    device_ = next(self.model.parameters()).device
                    inputs["input_ids"] = tokens["input_ids"].to(device_)
                    inputs["attention_mask"] = tokens["attention_mask"].to(device_)
                elif isinstance(seq_obj, torch.Tensor):
                    inputs["sequence_tokens"] = seq_obj
                else:
                    raise TypeError(f"[SaProtClassificationModel] Unexpected type under 'sequences': {type(seq_obj)}")
            # ---- STEP 1.2: ensure we actually have tokens
            if "input_ids" not in inputs and "sequence_tokens" not in inputs:
                raise ValueError(
                    "[SaProtClassificationModel] ESMC forward requires tokenized tensors "
                    "under 'input_ids' or 'sequence_tokens'."
                )
            # ---- STEP 1.3: unify field names
            if "sequence_tokens" not in inputs and "input_ids" in inputs:
                inputs["sequence_tokens"] = inputs["input_ids"]
            outputs = self.model(**inputs)
            return outputs.get("logits") if isinstance(outputs, dict) else outputs
        # ==============================================================
        # add esmc end 
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
            outputs = self.model(**inputs)
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