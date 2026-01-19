"""
Unsloth Trainer for Tool Calling SLMs
=====================================
Fine-tunes small language models for tool calling using Unsloth + LoRA.

Designed to run in Google Colab with T4 GPU.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# Supported non-gated SLMs for tool calling
# 
# How to choose:
#   - qwen2.5-3b (default): Best for tool calling, most accurate
#   - qwen2.5-1.5b: Faster, still good accuracy
#   - qwen2.5-0.5b: Fastest, for edge/mobile, basic tasks only
#   - phi-3.5-mini: Long context (128K), good for complex reasoning
#   - smollm2-1.7b: Smallest, fastest inference
#
SUPPORTED_MODELS = {
    "qwen2.5-3b": {
        "name": "Qwen2.5-3B-Instruct",
        "unsloth_id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "size": "3B",
        "context_length": 32768,
        "tool_calling": "excellent",
    },
    "qwen2.5-1.5b": {
        "name": "Qwen2.5-1.5B-Instruct",
        "unsloth_id": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "size": "1.5B",
        "context_length": 32768,
        "tool_calling": "good",
    },
    "qwen2.5-0.5b": {
        "name": "Qwen2.5-0.5B-Instruct",
        "unsloth_id": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "size": "0.5B",
        "context_length": 32768,
        "tool_calling": "basic",
    },
    "phi-3.5-mini": {
        "name": "Phi-3.5-mini-instruct",
        "unsloth_id": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "size": "3.8B",
        "context_length": 128000,
        "tool_calling": "good",
    },
    "smollm2-1.7b": {
        "name": "SmolLM2-1.7B-Instruct",
        "unsloth_id": "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
        "size": "1.7B",
        "context_length": 8192,
        "tool_calling": "basic",
    },
}

DEFAULT_MODEL = "qwen2.5-3b"


@dataclass
class TrainerConfig:
    """Configuration for fine-tuning."""
    
    # Model settings
    base_model: str = DEFAULT_MODEL  # Key from SUPPORTED_MODELS or full unsloth ID
    
    # LoRA settings (auto-adjusted based on dataset size if None)
    lora_rank: int = None  # Auto: 8 for small, 16 for medium, 32 for large
    lora_alpha: int = None  # Auto: same as rank
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training settings (auto-adjusted if None)
    epochs: int = None  # Auto: 5 for small, 3 for medium, 2 for large
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = None  # Auto: based on dataset size
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1  # 10% warmup
    weight_decay: float = 0.01
    
    # Output settings
    output_dir: str = "./trained_model"
    save_gguf: bool = True
    gguf_quantization: str = "q4_k_m"  # Good balance of size/quality
    
    def get_model_id(self) -> str:
        """Get the Unsloth model ID."""
        if self.base_model in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[self.base_model]["unsloth_id"]
        # Assume it's a full unsloth ID
        return self.base_model
    
    def auto_adjust_for_dataset(self, num_examples: int) -> "TrainerConfig":
        """
        Auto-adjust hyperparameters based on dataset size.
        
        Research-backed recommendations:
        - Small (<100): Conservative LR, more epochs, lower rank
        - Medium (100-500): Balanced settings
        - Large (500+): Can be more aggressive
        """
        if num_examples < 100:
            # Small dataset: conservative to avoid overfitting
            defaults = {
                "lora_rank": 8,
                "lora_alpha": 16,  # 2x rank for small data
                "epochs": 5,
                "learning_rate": 1e-4,  # Lower LR
            }
            size_category = "small"
        elif num_examples < 500:
            # Medium dataset: balanced
            defaults = {
                "lora_rank": 16,
                "lora_alpha": 16,
                "epochs": 3,
                "learning_rate": 2e-4,
            }
            size_category = "medium"
        else:
            # Large dataset: can be more aggressive
            defaults = {
                "lora_rank": 32,
                "lora_alpha": 32,
                "epochs": 2,
                "learning_rate": 2e-4,
            }
            size_category = "large"
        
        # Apply defaults only if not explicitly set
        if self.lora_rank is None:
            self.lora_rank = defaults["lora_rank"]
        if self.lora_alpha is None:
            self.lora_alpha = defaults["lora_alpha"]
        if self.epochs is None:
            self.epochs = defaults["epochs"]
        if self.learning_rate is None:
            self.learning_rate = defaults["learning_rate"]
        
        print(f"   📊 Dataset size: {num_examples} ({size_category})")
        print(f"   🔧 Auto-tuned: rank={self.lora_rank}, epochs={self.epochs}, lr={self.learning_rate}")
        
        return self


class UnslothTrainer:
    """
    Fine-tunes SLMs for tool calling using Unsloth.
    
    Usage:
        trainer = UnslothTrainer(
            training_data_path="training_data.jsonl",
            system_prompt="You are a calendar assistant...",
            config=TrainerConfig()
        )
        trainer.train()
        trainer.save()
    """
    
    def __init__(
        self,
        training_data_path: str,
        system_prompt: str,
        tools: list = None,
        config: TrainerConfig = None,
    ):
        self.training_data_path = training_data_path
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.config = config or TrainerConfig()
        
        # Will be set after loading
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Check if we're in Colab/have GPU
        self._check_environment()
    
    def _check_environment(self):
        """Check if we have the right environment for training."""
        try:
            import torch
            if not torch.cuda.is_available():
                print("⚠️ WARNING: No GPU detected!")
                print("   Training requires a GPU (Colab T4 recommended).")
                print("   You can still prepare the data and config.")
                self.has_gpu = False
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
                self.has_gpu = True
        except ImportError:
            print("⚠️ PyTorch not installed. Install with: pip install torch")
            self.has_gpu = False
    
    def _load_training_data(self) -> list:
        """Load and format training data from JSONL."""
        examples = []
        with open(self.training_data_path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        print(f"📊 Loaded {len(examples)} training examples")
        return examples
    
    def _format_for_qwen(self, example: dict) -> dict:
        """
        Format a training example for Qwen's chat template with tool calling.
        
        Qwen uses special tokens for tool calls:
        <|im_start|>user
        Query here
        <|im_end|>
        <|im_start|>assistant
        <tool_call>{"name": "tool", "arguments": {...}}</tool_call>
        <|im_end|>
        """
        messages = []
        
        # System message with tools
        system_content = self.system_prompt
        if self.tools:
            tools_json = json.dumps([t.to_dict() if hasattr(t, 'to_dict') else t for t in self.tools], indent=2)
            system_content += f"\n\nAvailable tools:\n{tools_json}"
        
        messages.append({"role": "system", "content": system_content})
        
        # User message
        query = example.get("query", example.get("user", ""))
        messages.append({"role": "user", "content": query})
        
        # Assistant response
        if "tool_call" in example:
            # Tool call response
            tool_call = example["tool_call"]
            tool_name = tool_call.get("tool", tool_call.get("name", ""))
            tool_args = tool_call.get("parameters", tool_call.get("arguments", {}))
            
            # Qwen tool call format
            assistant_content = f'<tool_call>{{"name": "{tool_name}", "arguments": {json.dumps(tool_args)}}}</tool_call>'
        else:
            # Regular response (no tool)
            assistant_content = example.get("response", example.get("assistant", ""))
        
        messages.append({"role": "assistant", "content": assistant_content})
        
        return {"messages": messages}
    
    def _prepare_dataset(self, examples: list):
        """Prepare dataset for training."""
        from datasets import Dataset
        
        formatted = [self._format_for_qwen(ex) for ex in examples]
        dataset = Dataset.from_list(formatted)
        
        print(f"✅ Prepared dataset with {len(dataset)} examples")
        return dataset
    
    def load_model(self, num_examples: int = None):
        """
        Load the base model with Unsloth.
        
        Args:
            num_examples: If provided, auto-adjusts LoRA params based on dataset size
        """
        if not self.has_gpu:
            raise RuntimeError("GPU required for training. Please run in Colab.")
        
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth not installed. Install with:\n"
                "pip install unsloth\n"
                "Or run in Google Colab with GPU."
            )
        
        # Auto-adjust if we know dataset size
        if num_examples:
            self.config.auto_adjust_for_dataset(num_examples)
        else:
            # Use defaults if not auto-adjusted
            if self.config.lora_rank is None:
                self.config.lora_rank = 16
            if self.config.lora_alpha is None:
                self.config.lora_alpha = 16
        
        model_id = self.config.get_model_id()
        print(f"🔄 Loading model: {model_id}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        # Apply LoRA
        print(f"🔧 Applying LoRA adapters (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        print("✅ Model loaded with LoRA")
        return self.model, self.tokenizer
    
    def train(self):
        """Run the fine-tuning process."""
        if self.model is None:
            self.load_model()
        
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import is_bfloat16_supported
        
        # Load and prepare data
        examples = self._load_training_data()
        
        # Auto-adjust hyperparameters based on dataset size
        self.config.auto_adjust_for_dataset(len(examples))
        
        dataset = self._prepare_dataset(examples)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=10,
            save_strategy="epoch",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            seed=42,
        )
        
        # Create trainer
        print("🚀 Starting training...")
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=self.config.max_seq_length,
        )
        
        # Train!
        self.trainer.train()
        print("✅ Training complete!")
        
        return self.trainer
    
    def save(self) -> dict:
        """
        Save the trained model.
        
        Returns dict with paths to saved files.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Run train() first.")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        saved_paths = {}
        
        # Save LoRA adapters
        lora_path = os.path.join(self.config.output_dir, "lora_adapters")
        print(f"💾 Saving LoRA adapters to {lora_path}...")
        self.model.save_pretrained(lora_path)
        self.tokenizer.save_pretrained(lora_path)
        saved_paths["lora"] = lora_path
        
        # Save merged model (optional, takes more space)
        merged_path = os.path.join(self.config.output_dir, "merged_model")
        print(f"💾 Saving merged model to {merged_path}...")
        self.model.save_pretrained_merged(merged_path, self.tokenizer)
        saved_paths["merged"] = merged_path
        
        # Save GGUF for Ollama
        if self.config.save_gguf:
            gguf_path = os.path.join(self.config.output_dir, "gguf")
            print(f"💾 Converting to GGUF ({self.config.gguf_quantization})...")
            self.model.save_pretrained_gguf(
                gguf_path,
                self.tokenizer,
                quantization_method=self.config.gguf_quantization,
            )
            saved_paths["gguf"] = gguf_path
            print(f"✅ GGUF saved! Use with Ollama:")
            print(f"   ollama create my-agent -f {gguf_path}/Modelfile")
        
        print("\n✅ All models saved!")
        return saved_paths
    
    def test(self, query: str) -> str:
        """Test the trained model with a query."""
        if self.model is None:
            raise RuntimeError("No model loaded. Run train() first.")
        
        from unsloth import FastLanguageModel
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Format input
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        
        # Generate
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            use_cache=True,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response


def get_training_notebook_code(
    training_data_path: str,
    system_prompt: str,
    tools: list,
    config: TrainerConfig = None,
) -> str:
    """
    Generate Python code for a Colab training notebook.
    
    This is used by the Meta-Agent Backend to create
    pre-configured notebooks for users.
    """
    config = config or TrainerConfig()
    tools_json = json.dumps([t.to_dict() if hasattr(t, 'to_dict') else t for t in tools], indent=2)
    
    code = f'''# OnsetLab Agent Training
# =========================
# Auto-generated training notebook

# 1. Install dependencies
!pip install -q unsloth transformers datasets trl

# 2. Configuration (auto-generated from your settings)
TRAINING_DATA = """{training_data_path}"""
SYSTEM_PROMPT = """{system_prompt}"""
TOOLS = {tools_json}

BASE_MODEL = "{config.base_model}"
LORA_RANK = {config.lora_rank}
EPOCHS = {config.epochs}
OUTPUT_DIR = "{config.output_dir}"

# 3. Run training
from onsetlab.training import UnslothTrainer, TrainerConfig

config = TrainerConfig(
    base_model=BASE_MODEL,
    lora_rank=LORA_RANK,
    epochs=EPOCHS,
    output_dir=OUTPUT_DIR,
    save_gguf=True,
)

trainer = UnslothTrainer(
    training_data_path=TRAINING_DATA,
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
    config=config,
)

trainer.train()
saved = trainer.save()

print("\\n🎉 Training complete!")
print(f"GGUF model saved to: {{saved['gguf']}}")
print("\\nDownload the GGUF file and use with Ollama:")
print("  ollama create my-agent -f Modelfile")
'''
    return code
