import torch
import time
from pipeline import MultimodalEmotionPipeline

# (Your MultimodalEmotionPipeline class and generate_response method go here)

def run_streaming_simulation():
    pipeline = MultimodalEmotionPipeline()
    
    conversation_stream = [
        "Hey, did you look at the final project dashboard yet?",
        "Wait, are you serious? Everything we built just crashed?!",
        "Oh wow, never mind, it looks like it was just a temporary glitch. We're back online!"
    ]
    
    print("\nStarting Real-Time Multimodal Emotion Stream Simulation...")
    print("---------------------------------------------------------")
    
    for i, utterance in enumerate(conversation_stream, 1):
        start_time = time.time()
        response = pipeline.generate_response(utterance)
        latency = (time.time() - start_time) * 1000
        
        print(f"\n[Turn {i}]")
        print(f"Input Utterance: '{utterance}'")
        print(f"Model Output:\n{response}")
        print(f"Latency: {latency:.2f}ms")
        print("-" * 50)

if __name__ == "__main__":
    run_streaming_simulation()