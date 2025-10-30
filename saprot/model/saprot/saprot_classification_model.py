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

            model_ref = self.model

            # unwrap peft/adapter wrappers until we reach real backbone
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

            # If inside is an ESMC / EvolutionaryScale-type model
            if (
                "esmc" in real_cls_name
                or "evolutionaryscale" in real_cls_name
            ):
                if isinstance(inputs, dict):
                    # ensure tokens key exists
                    if "sequence_tokens" not in inputs:
                        if "input_ids" in inputs and inputs["input_ids"] is not None:
                            print("[DEBUG] Mapping input_ids -> sequence_tokens")
                            inputs["sequence_tokens"] = inputs.pop("input_ids")
                        elif "tokens" in inputs and inputs["tokens"] is not None:
                            print("[DEBUG] Mapping tokens -> sequence_tokens")
                            inputs["sequence_tokens"] = inputs.pop("tokens")
                        elif "sequences" in inputs and inputs["sequences"] is not None:
                            print("[DEBUG] Mapping sequences -> sequence_tokens")
                            inputs["sequence_tokens"] = inputs.pop("sequences")
                        else:
                            print(f"[ERROR] inputs keys={list(inputs.keys())}")
                            raise ValueError(
                                "[SaProtClassificationModel] Missing sequence_tokens "
                                "(input_ids/tokens/sequences all missing or None)"
                            )
                    else:
                        print("[DEBUG] sequence_tokens already present in inputs")

                # Print shape of the mapped sequence_tokens if available
                if "sequence_tokens" in inputs:
                    try:
                        print(f"[DEBUG] sequence_tokens shape: {inputs['sequence_tokens'].shape}")
                    except Exception as e:
                        print(f"[WARN] Cannot get shape of sequence_tokens: {e}")

                outputs = self.model(**inputs)
                print("[DEBUG] Forwarded through ESMC successfully")

                if isinstance(outputs, dict):
                    print(f"[DEBUG] Output keys: {list(outputs.keys())}")
                    return outputs.get("logits", list(outputs.values())[0])
                return outputs

            else:
                print("[DEBUG] Not an ESMC model, skipping remap")

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