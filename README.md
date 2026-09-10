    # Real-Time Multimodal ML Challenge - Submission

## Overview
This repository contains a local, end-to-end multimodal (Text + Audio) inference prototype built for the Human Computer Lab Machine Learning Intern position. The system processes conversational turns, evaluates acoustic and textual inputs, predicts MELD-aligned emotion tags, and generates context-aware character responses while remaining strictly under the 6-billion parameter limit.

## Architecture & Model Backbones
To maximize performance while strictly adhering to the **<6B parameter ceiling**, we utilized a modular two-component backbone:
1. **Text Backbone:** `Qwen/Qwen2-1.5B-Instruct` (~1.5 Billion parameters) — Handles semantic reasoning, instruction-following, and response generation.
2. **Audio Encoder:** `openai/whisper-tiny` (~39 Million parameters) — Extracts acoustic embeddings from raw speech waveforms.
3. **Fusion Mechanism:** A lightweight linear projection layer bridges the Whisper hidden dimensions into the Qwen embedding space.
* **Total System Parameter Footprint:** ~1.54B parameters (**100% compliant with the <6B limit**).

## Project Structure
- `audit_models.py`: Programmatically loads backbones, counts parameters, and verifies compliance.
- `pipeline.py`: Contains the core `MultimodalEmotionPipeline` class handling feature extraction and LLM generation.
- `inference.py`: Simulates a real-time conversational streaming loop with turn-by-turn latency tracking.

## Setup & Execution
1. Create and activate your virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1