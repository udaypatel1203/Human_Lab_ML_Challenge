import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq

def count_parameters(model):
    """Returns the total number of parameters (trainable and non-trainable) in a model."""
    return sum(p.numel() for p in model.parameters())

def main():
    print("Initializing model audit for Human Computer Lab ML Challenge...")
    
    # 1. Select backbones
    # Text Backbone: Qwen2-1.5B-Instruct (~1.5 billion parameters)
    text_model_name = "Qwen/Qwen2-1.5B-Instruct"
    
    # Audio Backbone: Whisper-tiny (~39 million parameters)
    audio_model_name = "openai/whisper-tiny"
    
    print(f"\nLoading Text Model: {text_model_name}...")
    # Load with low_cpu_mem_usage to speed up local initialization
    text_model = AutoModelForCausalLM.from_pretrained(
        text_model_name, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    )
    text_params = count_parameters(text_model)
    print(f"-> Text Model Parameters: {text_params:,} (~{text_params / 1e9:.2f}B)")

    print(f"\nLoading Audio Model: {audio_model_name}...")
    audio_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        audio_model_name, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    )
    audio_params = count_parameters(audio_model)
    print(f"-> Audio Model Parameters: {audio_params:,} (~{audio_params / 1e6:.2f}M)")

    # 2. Calculate Total Parameter Sum (Enforcing the <6B Rule)
    total_parameters = text_params + audio_params
    param_limit = 6_000_000_000  # 6 Billion limit
    
    print("\n" + "="*40)
    print(f"TOTAL SYSTEM PARAMETERS: {total_parameters:,} (~{total_parameters / 1e9:.2f}B)")
    print(f"CHALLENGE PARAMETER LIMIT: {param_limit:,} (6.0B)")
    print("="*40)

    if total_parameters < param_limit:
        print("[SUCCESS] Model configuration is STRICTLY compliant with the <6B parameter constraint!")
    else:
        print("[FAIL] Configuration exceeds the 6B parameter ceiling. Choose smaller backbones.")

if __name__ == "__main__":
    main()