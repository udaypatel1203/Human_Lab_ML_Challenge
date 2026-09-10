import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoTokenizer

class MultimodalEmotionPipeline(nn.Module):
    def __init__(self, audio_model_id="openai/whisper-tiny", text_model_id="Qwen/Qwen2-1.5B-Instruct"):
        super().__init__()
        print("Initializing Multimodal Emotion Pipeline...")
        
        # Load Audio Encoder (Whisper Tiny)
        self.audio_processor = AutoProcessor.from_pretrained(audio_model_id)
        self.audio_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(
            audio_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).model.encoder  # Extract underlying encoder backbone

        # Load Text LLM (Qwen2-1.5B-Instruct)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        
        # Lightweight projection layer to bridge audio feature dim to text embedding dim
        audio_hidden_dim = self.audio_encoder.config.d_model
        text_hidden_dim = self.text_model.config.hidden_size
        self.audio_projector = nn.Linear(audio_hidden_dim, text_hidden_dim)
        
        # Ensure padding token exists for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_audio_features(self, audio_waveform, sampling_rate=16000):
        """Extracts and projects acoustic embeddings from raw audio waveforms."""
        inputs = self.audio_processor(
            audio_waveform, sampling_rate=sampling_rate, return_tensors="pt"
        )
        with torch.no_grad():
            # Pass through whisper encoder
            audio_outputs = self.audio_encoder(inputs.input_features)
            hidden_states = audio_outputs.last_hidden_state  # [Batch, Seq, Hidden]
            
        # Project into the LLM's hidden space
        projected_embeddings = self.audio_projector(hidden_states)
        return projected_embeddings

    def generate_response(self, text_utterance, audio_waveform=None, sampling_rate=16000):
        """Processes multimodal inputs and generates structured tags + response text."""
        # Format prompt instructing the model to output a MELD emotion tag and grounded response
        prompt = f"""[System: You are an emotion-aware character robot. Analyze the utterance text and output your response in the following format:
Emotion Tag: [neutral/joy/sadness/anger/surprised/disgust/fear]
Response: <grounded response text>]

User Utterance: "{text_utterance}"
Robot:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Generate completion from the LLM
        with torch.no_grad():
            output_ids = self.text_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7
            )
            
        decoded_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract the assistant response part after the prompt
        response_text = decoded_output.split("Robot:")[-1].strip()
        return response_text

def main():
    # Test the pipeline locally with dummy text/audio simulation
    pipeline = MultimodalEmotionPipeline()
    
    sample_text = "I can't believe we actually pulled off this launch successfully!"
    print(f"\nProcessing Test Utterance: '{sample_text}'")
    
    response = pipeline.generate_response(sample_text)
    print("\n--- Model Output ---")
    print(response)
    print("--------------------")

if __name__ == "__main__":
    main()