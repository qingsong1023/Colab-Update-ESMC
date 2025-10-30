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
        is_esmc_model = False
        model_ref = self.model

        if model_ref.__class__.__name__.lower().startswith("esmc"):
            is_esmc_model = True
        elif hasattr(model_ref, "model") and model_ref.model.__class__.__name__.lower().startswith("esmc"):
            is_esmc_model = True

        if is_esmc_model:
            if isinstance(inputs, dict):
                if "input_ids" in inputs and "tokens" not in inputs:
                    inputs["tokens"] = inputs.pop("input_ids")
            outputs = self.model(**inputs)
            return outputs

        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        outputs = self.model(inputs)
        return outputs

        # ---------------------------------------------
        # 1️如果冻结 backbone ：直接取 embedding 平均值
        # ---------------------------------------------
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)
            return logits

        # ---------------------------------------------
        # 2️检测模型类型
        # ---------------------------------------------
        # ESMC 分支
        if isinstance(getattr(self, "model", None), object) and (
            "ESMC" in self.model.__class__.__name__ or hasattr(self.model, "embed_dim")
        ):
            # 调用 EvolutionaryScale 的 forward 接口
            # 通常输入为蛋白质序列的 token 序列、或 embeddings，取决于上游包装
            from esm.sdk.api import LogitsConfig

            logits_cfg = LogitsConfig(sequence=True)
            outputs = self.model.forward(inputs, logits_cfg)
            # outputs 是 dict，例如 {"logits": tensor, "probabilities": tensor, ...}
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                raise ValueError(f"Unexpected ESMC output type: {type(outputs)}")

            return logits

        # ---------------------------------------------
        # 3️普通 HuggingFace 模型分支
        # ---------------------------------------------
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logits

    def loss_func(self, stage, logits, labels):
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