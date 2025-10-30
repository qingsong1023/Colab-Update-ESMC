# ===============================================================
# ESMC specific LoRA Adapter (lora_esmc_adapter.py)
# ===============================================================

import torch
from peft import LoraConfig, get_peft_model


def apply_lora_to_esmc(
    model,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
    trainable: bool = True,
    adapter_name: str = "default",
):
    """
    Apply a LoRA adapter specifically for EvolutionaryScale's ESMC models.
    Independent from Hugging Face AutoModel architecture no SafePatch needed.

    Args:
        model:          ESMC backbone instance (e.g., ESMC.from_pretrained("esmc_300m"))
        r:              LoRA bottleneck dimension
        alpha:          LoRA scaling parameter
        dropout:        LoRA dropout probability
        bias:           Bias option ('none' | 'all' | 'lora_only')
        trainable:      Whether LoRA layers are trainable
        adapter_name:   Name of the LoRA adapter

    Returns:
        The modified model with LoRA layers injected.
    """
    print("[LoRA][ESMC] Injecting LoRA modules into ESMC...")

    # 1. Define target modules specific to the ESMC architecture
    target_modules = [
        "q_proj", "k_proj", "v_proj", "out_proj",  # Attention layers
        "fc1", "fc2", "mlp"                        # Feed‑forward layers
    ]

    # 2. Create a LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        task_type="FEATURE_EXTRACTION",  # Suitable for representation models like ESMC
        target_modules=target_modules,
        inference_mode=not trainable,
        init_lora_weights="gaussian",
    )

    # 3. Inject LoRA layers
    model = get_peft_model(model, lora_config, adapter_name=adapter_name)

    # 4. Show summary
    model.print_trainable_parameters()
    print(f"[LoRA][ESMC] LoRA injection completed. Active adapter: {adapter_name}")

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