# ===============================================================
# ESMC specific LoRA Adapter (lora_esmc_adapter.py)
# ===============================================================

import torch
from peft import LoraConfig, get_peft_model


def apply_lora_to_esmc(model, lora_kwargs, num_lora=1, is_trainable=False, config_list=None):
    """
    Apply LoRA to ESMC backbone (EvolutionaryScale models).
    """
    print("Injecting LoRA into ESMC model...")

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "mlp"]

    r = getattr(lora_kwargs, "r", 8)
    lora_dropout = getattr(lora_kwargs, "lora_dropout", 0.1)
    lora_alpha = getattr(lora_kwargs, "lora_alpha", 16)

    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        target_modules=target_modules,
        modules_to_save=["classifier"],
        inference_mode=False,
        r=r,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("LoRA successfully applied to ESMC backbone.")
    return model


# ---------------------------------------------------------------
# Example: interactive ESMC + LoRA injection (for Jupyter / Colab)
# ---------------------------------------------------------------
if __name__ == "__main__":
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    import pretrained  # Replace with your local ESMC loading SDK

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_dropdown = widgets.Dropdown(
        options=list(LOCAL_MODEL_REGISTRY.keys()),
        description="Models:",
        layout=widgets.Layout(width="300px"),
    )
    load_button = widgets.Button(description="Load model")
    output_area = widgets.Output()
    display(model_dropdown, load_button, output_area)

    def on_load_clicked(b):
        with output_area:
            clear_output()
            model_name = model_dropdown.value
            print(f"Loading ESMC backbone: {model_name}")

            # Load the base ESMC model
            base_model = pretrained.load_local_model(model_name).to(device)
            base_model.eval()

            # Apply ESMspecific LoRA
            from lora_esmc_adapter import apply_lora_to_esmc
            esm_model = apply_lora_to_esmc(base_model, r=8, alpha=32, dropout=0.05)

            # 3Quick forward check
            with torch.no_grad():
                dummy_tokens = torch.randint(0, 30, (1, 16), device=device)
                try:
                    out = esm_model(sequence_tokens=dummy_tokens)
                    shape = out[0].shape if isinstance(out, tuple) else out.shape
                    print("Output tensor shape:", shape)
                except Exception as e:
                    print("[Warning] Forward pass failed. Possibly adjust input args.", e)

            print("ESMC LoRA initialization complete.")

    load_button.on_click(on_load_clicked)