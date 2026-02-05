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


# =============================================================================
# Model Configuration
# =============================================================================
# Supported models for tool-calling fine-tuning.
# 
# ToolLLaMA: Pre-trained on 16K+ APIs (ToolBench), already knows tool patterns
# NexusRaven: Optimized for function calling, surpasses GPT-4 on benchmarks
# Qwen: General purpose, needs more training data for tools
#
SUPPORTED_MODELS = {
    # RECOMMENDED: Pre-trained on function calling (BEST for small size)
    "phi-3.5-fc": {
        "name": "Phi-3.5-mini-instruct-hermes-fc",
        "unsloth_id": "unsloth/Phi-3.5-mini-instruct",  # Use base for Unsloth
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "size": "3.8B",
        "context_length": 128000,
        "tool_format": "hermes",  # Uses Hermes function calling format
        "pretrained_on_tools": True,  # Pre-trained on function calling!
    },
    "toolllama-7b": {
        "name": "ToolLLaMA-2-7b-v2",
        "unsloth_id": "ToolBench/ToolLLaMA-2-7b-v2",
        "hf_id": "ToolBench/ToolLLaMA-2-7b-v2",
        "size": "7B",
        "context_length": 4096,
        "tool_format": "toolllama",  # Uses ToolBench format
        "pretrained_on_tools": True,
    },
    "nexusraven-13b": {
        "name": "NexusRaven-V2-13B",
        "unsloth_id": "Nexusflow/NexusRaven-V2-13B",
        "hf_id": "Nexusflow/NexusRaven-V2-13B",
        "size": "13B",
        "context_length": 16384,
        "tool_format": "nexusraven",  # Uses Python function signatures
        "pretrained_on_tools": True,
    },
    # General purpose (need more training data)
    "qwen2.5-3b": {
        "name": "Qwen2.5-3B-Instruct",
        "unsloth_id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "size": "3B",
        "context_length": 32768,
        "tool_format": "qwen",
        "pretrained_on_tools": False,
    },
    "qwen2.5-7b": {
        "name": "Qwen2.5-7B-Instruct",
        "unsloth_id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "size": "7B",
        "context_length": 131072,
        "tool_format": "qwen",
        "pretrained_on_tools": False,
    },
}

# Default to Phi-3.5-FC - best small model for function calling
DEFAULT_MODEL = "phi-3.5-fc"
MODEL_CONFIG = SUPPORTED_MODELS[DEFAULT_MODEL]


@dataclass
class TrainerConfig:
    """Configuration for fine-tuning. Supports phi-3.5-fc (recommended), qwen2.5-3b and qwen2.5-7b."""
    
    # Model settings: "phi-3.5-fc" (RECOMMENDED), "qwen2.5-3b" (fast), or "qwen2.5-7b"
    base_model: str = DEFAULT_MODEL
    
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
        # Check by key first (e.g., "qwen2.5-3b")
        if self.base_model in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[self.base_model]["unsloth_id"]
        
        # Check by display name (e.g., "Qwen2.5-3B-Instruct")
        for key, info in SUPPORTED_MODELS.items():
            if info["name"] == self.base_model:
                return info["unsloth_id"]
        
        # Check if it starts with "unsloth/" (already a valid ID)
        if self.base_model.startswith("unsloth/"):
            return self.base_model
        
        # Try adding "unsloth/" prefix
        return f"unsloth/{self.base_model}"
    
    def auto_adjust_for_dataset(self, num_examples: int, num_tools: int = None) -> "TrainerConfig":
        """
        Auto-adjust hyperparameters based on dataset size and tool count.
        
        Research-backed recommendations:
        - Small (<100): Conservative LR, more epochs, lower rank
        - Medium (100-500): Balanced settings
        - Large (500-1500): Slightly higher rank, fewer epochs
        
        Tool count also affects rank (capped at 32 given max 1500 examples):
        - ≤15 tools: No boost needed
        - >15 tools: Small boost (+4 per 5 tools over 15)
        
        Max rank is 32 because:
        - Data generator caps at 1500 examples
        - Tool calling is structured (not creative)
        - Higher rank = overfitting risk with limited data
        """
        if num_examples < 100:
            # Small dataset: conservative to avoid overfitting
            defaults = {
                "lora_rank": 8,
                "lora_alpha": 16,  # 2x rank for small data (more regularization)
                "epochs": 5,
                "learning_rate": 1e-4,  # Lower LR
            }
            size_category = "small"
        elif num_examples < 500:
            # Medium dataset: slightly more epochs for better generalization
            # 3B models need more passes on medium data to learn patterns reliably
            defaults = {
                "lora_rank": 16,
                "lora_alpha": 16,
                "epochs": 4,  # Bumped from 3 to 4 for better tool+conversation balance
                "learning_rate": 1.5e-4,  # Slightly lower for better generalization
            }
            size_category = "medium"
        else:
            # Large dataset (500-1500): can use higher rank
            defaults = {
                "lora_rank": 24,  # Start at 24 for large (not 32)
                "lora_alpha": 24,
                "epochs": 2,
                "learning_rate": 2e-4,
            }
            size_category = "large"
        
        # Adjust rank based on tool count (more tools = more complexity)
        # Only boost if >15 tools, max rank is 32
        if num_tools is not None and num_tools > 15:
            # Conservative boost: +4 per 5 tools over 15
            rank_boost = min(8, (num_tools - 15) // 5 * 4)
            defaults["lora_rank"] = min(32, defaults["lora_rank"] + rank_boost)
            defaults["lora_alpha"] = defaults["lora_rank"]
        
        # Apply defaults only if not explicitly set
        if self.lora_rank is None:
            self.lora_rank = defaults["lora_rank"]
        if self.lora_alpha is None:
            self.lora_alpha = defaults["lora_alpha"]
        if self.epochs is None:
            self.epochs = defaults["epochs"]
        if self.learning_rate is None:
            self.learning_rate = defaults["learning_rate"]
        
        tool_info = f", {num_tools} tools" if num_tools else ""
        print(f"   📊 Dataset size: {num_examples} ({size_category}{tool_info})")
        print(f"   🔧 Auto-tuned: rank={self.lora_rank}, epochs={self.epochs}, lr={self.learning_rate}")
        
        return self


class UnslothTrainer:
    """
    Fine-tunes SLMs for tool calling using Unsloth.
    
    Usage:
        trainer = UnslothTrainer(
            training_data_path="train.jsonl",
            validation_data_path="validation.jsonl",  # Optional
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
        validation_data_path: str = None,  # NEW: Use pre-split validation
    ):
        self.training_data_path = training_data_path
        self.validation_data_path = validation_data_path
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
        """Load training data from JSONL."""
        return self._load_data_from_path(self.training_data_path)
    
    def _load_data_from_path(self, path: str) -> list:
        """Load data from a JSONL file."""
        examples = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples
    
    def _format_example_to_text(self, example: dict) -> str:
        """
        Convert an example to text using tokenizer's chat template.
        
        Handles:
        - New format: {"messages": [...]} with role: system/user/assistant/tool
        - Old format: {"query": ..., "tool_call": {...}}
        
        Multi-turn format is fully supported (role: tool messages).
        """
        # Check if already in messages format
        if "messages" in example:
            messages = example["messages"]
        else:
            # Build messages from old format (backward compatibility)
            messages = self._convert_old_format(example)
        
        # Use tokenizer's chat template
        # This handles Qwen's special tokens correctly
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text
    
    def _convert_old_format(self, example: dict) -> list:
        """Convert old format to messages format (backward compatibility)."""
        messages = []
        
        # System message
        messages.append({"role": "system", "content": self.system_prompt})
        
        # User message
        query = example.get("query", example.get("user", ""))
        messages.append({"role": "user", "content": query})
        
        # Assistant response
        if "tool_call" in example:
            tool_call = example["tool_call"]
            tool_name = tool_call.get("tool", tool_call.get("name", ""))
            tool_args = tool_call.get("parameters", tool_call.get("arguments", {}))
            # Use our format: {"tool": ..., "parameters": ...}
            assistant_content = f'<tool_call>\n{json.dumps({"tool": tool_name, "parameters": tool_args})}\n</tool_call>'
        else:
            assistant_content = example.get("response", example.get("assistant", ""))
        
        messages.append({"role": "assistant", "content": assistant_content})
        
        return messages
    
    def load_model(self, num_examples: int = None, num_tools: int = None):
        """
        Load the base model.
        
        Uses Unsloth for optimized models (unsloth/*), standard transformers for others.
        
        Args:
            num_examples: If provided, auto-adjusts LoRA params based on dataset size
            num_tools: If provided, adjusts rank based on tool count
        """
        if not self.has_gpu:
            raise RuntimeError("GPU required for training. Please run in Colab.")
        
        # Ensure LoRA config is set
        if self.config.lora_rank is None or self.config.lora_alpha is None:
            if num_examples:
                self.config.auto_adjust_for_dataset(num_examples, num_tools)
            else:
                if self.config.lora_rank is None:
                    self.config.lora_rank = 16
                if self.config.lora_alpha is None:
                    self.config.lora_alpha = 16
        
        model_id = self.config.get_model_id()
        print(f"🔄 Loading model: {model_id}")
        
        # Check if this is an Unsloth-optimized model
        is_unsloth_model = model_id.startswith("unsloth/")
        
        if is_unsloth_model:
            # Use Unsloth's FastLanguageModel for optimized loading
            self._load_with_unsloth(model_id)
        else:
            # Use standard transformers + PEFT for other models (like ToolLLaMA)
            self._load_with_transformers(model_id)
        
        print("✅ Model loaded with LoRA")
        return self.model, self.tokenizer
    
    def _load_with_unsloth(self, model_id: str):
        """Load model using Unsloth's optimized loader."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth not installed. Install with:\n"
                "pip install unsloth\n"
                "Or run in Google Colab with GPU."
            )
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
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
        self._use_unsloth = True
    
    def _load_with_transformers(self, model_id: str):
        """Load model using standard transformers + PEFT (for non-Unsloth models like ToolLLaMA)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        print(f"   Using standard transformers loading for {model_id}")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # Required for causal LM training
        
        # Set chat template for LLaMA-2 models (they don't have one by default)
        if self.tokenizer.chat_template is None:
            print("   Setting LLaMA-2 chat template...")
            # Flexible LLaMA-2 chat template (no strict alternation - works with Observation messages)
            self.tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set loop_messages = messages[1:] %}{% else %}{% set system_message = '' %}{% set loop_messages = messages %}{% endif %}{% if system_message %}{{ bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' }}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{% if loop.index0 == 0 and system_message %}{{ message['content'].strip() + ' [/INST]' }}{% else %}{{ '[INST] ' + message['content'].strip() + ' [/INST]' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'].strip() + ' ' + eos_token }}{% endif %}{% endfor %}"""
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        print(f"🔧 Applying LoRA adapters (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})...")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self._use_unsloth = False
    
    def train(self):
        """Run the fine-tuning process."""
        # Disable Weights & Biases prompting
        import os
        import torch
        os.environ["WANDB_DISABLED"] = "true"
        
        from datasets import Dataset
        
        # Load training data FIRST (need size for auto-adjustment)
        train_examples = self._load_training_data()
        
        # Load validation data (if provided separately)
        val_examples = None
        if self.validation_data_path:
            val_examples = self._load_data_from_path(self.validation_data_path)
            print(f"📊 Loaded {len(val_examples)} validation examples")
        
        # Auto-adjust hyperparameters BEFORE loading model
        # (LoRA rank/alpha need to be set before applying adapters)
        num_tools = len(self.tools) if self.tools else None
        self.config.auto_adjust_for_dataset(len(train_examples), num_tools)
        
        # NOW load model with correct LoRA settings
        if self.model is None:
            self.load_model(num_examples=len(train_examples), num_tools=num_tools)
        
        # Format all examples into text strings
        print("📝 Formatting training data...")
        train_texts = [self._format_example_to_text(ex) for ex in train_examples]
        
        # Show sample
        if train_texts:
            print(f"\n📝 Sample (first 500 chars):\n{train_texts[0][:500]}...")
        
        # Create training dataset
        train_dataset = Dataset.from_dict({"text": train_texts})
        
        # Handle validation data
        if val_examples:
            # Use provided validation data (preserves stratification!)
            val_texts = [self._format_example_to_text(ex) for ex in val_examples]
            val_dataset = Dataset.from_dict({"text": val_texts})
            print(f"\n✅ Using pre-split data (stratified):")
        else:
            # Fall back to splitting if no validation file provided
            split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset["train"]
            val_dataset = split_dataset["test"]
            print(f"\n✅ Auto-split data (90/10):")
        
        print(f"   📊 Training: {len(train_dataset)} examples")
        print(f"   📊 Validation: {len(val_dataset)} examples")
        
        # Safety check: ensure all training params are set (should be done by auto_adjust)
        if self.config.epochs is None:
            self.config.epochs = 3  # Safe default
        if self.config.learning_rate is None:
            self.config.learning_rate = 2e-4  # Safe default
        
        # Check for bfloat16 support
        def check_bf16_support():
            """Check if GPU supports bfloat16."""
            if not torch.cuda.is_available():
                return False
            # Check compute capability (Ampere+ GPUs support bf16)
            major, _ = torch.cuda.get_device_capability()
            return major >= 8  # Ampere (A100, etc.) and newer
        
        use_bf16 = check_bf16_support()
        
        print("🚀 Starting training (with validation)...")
        print(f"   Using {'bf16' if use_bf16 else 'fp16'} precision")
        
        # Use standard Trainer (stable API, no SFTTrainer compatibility issues)
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        # Tokenize the datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=self.config.max_seq_length, 
                padding="max_length"
            )
        
        print("   Tokenizing datasets...")
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",  # New API name
            fp16=not use_bf16,
            bf16=use_bf16,
            optim="adamw_8bit",
            seed=42,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        # Train!
        train_result = self.trainer.train()
        
        # Print final loss comparison
        print("\n" + "=" * 50)
        print("📊 Training Summary")
        print("=" * 50)
        
        # Get final metrics (handle different trl versions)
        final_train_loss = getattr(train_result, 'training_loss', None)
        if final_train_loss is None:
            # Try alternative attribute names
            metrics = getattr(train_result, 'metrics', {})
            final_train_loss = metrics.get('train_loss', metrics.get('loss', 0))
        print(f"   Final Training Loss: {final_train_loss:.4f}")
        
        # Run final evaluation
        try:
            eval_metrics = self.trainer.evaluate()
            final_val_loss = eval_metrics.get("eval_loss", 0)
            print(f"   Final Validation Loss: {final_val_loss:.4f}")
            
            # Check for overfitting
            if final_val_loss > final_train_loss * 1.5:
                print(f"   ⚠️ Warning: Possible overfitting (val_loss >> train_loss)")
            elif final_val_loss < final_train_loss * 1.2:
                print(f"   ✅ Good fit: validation and training loss are close")
        except Exception as e:
            print(f"   ⚠️ Could not run evaluation: {e}")
        
        print("=" * 50)
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
    
    def evaluate(self, test_data_path: str) -> dict:
        """
        Evaluate model on held-out test set.
        
        Measures tool prediction accuracy:
        - Did the model call the correct tool?
        - Did it avoid hallucinating tools not in training?
        
        Args:
            test_data_path: Path to test.jsonl (from data generator)
        
        Returns:
            dict with:
            - accuracy: Overall % correct
            - correct: Number of correct predictions
            - total: Total test examples
            - per_tool: Accuracy breakdown by tool
            - errors: List of incorrect predictions for analysis
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Run train() first.")
        
        from unsloth import FastLanguageModel
        import re
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Load test data
        test_examples = self._load_data_from_path(test_data_path)
        
        print(f"\n📊 Evaluating on {len(test_examples)} test examples...")
        
        # Track results
        correct = 0
        total = 0
        per_tool = {}  # tool -> {"correct": x, "total": y}
        errors = []
        
        for i, example in enumerate(test_examples):
            messages = example.get("messages", [])
            
            # Find user query and expected tool
            user_query = None
            expected_tool = None
            
            for msg in messages:
                if msg["role"] == "user":
                    user_query = msg["content"]
                elif msg["role"] == "assistant":
                    # Extract expected tool from assistant response
                    content = msg["content"]
                    if "<tool_call>" in content or '"tool"' in content:
                        # Try to parse tool name
                        match = re.search(r'"tool"\s*:\s*"([^"]+)"', content)
                        if match:
                            expected_tool = match.group(1)
                            break
            
            # Skip edge cases (no tool expected)
            if not user_query or not expected_tool:
                continue
            
            total += 1
            
            # Initialize per-tool tracking
            if expected_tool not in per_tool:
                per_tool[expected_tool] = {"correct": 0, "total": 0}
            per_tool[expected_tool]["total"] += 1
            
            # Generate prediction
            try:
                prediction = self.test(user_query)
                
                # Extract predicted tool
                predicted_tool = None
                if "<tool_call>" in prediction or '"tool"' in prediction:
                    match = re.search(r'"tool"\s*:\s*"([^"]+)"', prediction)
                    if match:
                        predicted_tool = match.group(1)
                
                # Check if correct
                if predicted_tool == expected_tool:
                    correct += 1
                    per_tool[expected_tool]["correct"] += 1
                else:
                    errors.append({
                        "query": user_query[:100],
                        "expected": expected_tool,
                        "predicted": predicted_tool,
                    })
                    
            except Exception as e:
                errors.append({
                    "query": user_query[:100],
                    "expected": expected_tool,
                    "error": str(e),
                })
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   ... {i + 1}/{len(test_examples)}")
        
        # Calculate accuracy
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Calculate per-tool accuracy
        tool_accuracy = {}
        for tool, stats in per_tool.items():
            if stats["total"] > 0:
                tool_accuracy[tool] = {
                    "accuracy": stats["correct"] / stats["total"] * 100,
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Evaluation Results")
        print("=" * 50)
        print(f"   Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"\n   Per-Tool Breakdown:")
        for tool, stats in sorted(tool_accuracy.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"      {tool}: {stats['accuracy']:.0f}% ({stats['correct']}/{stats['total']})")
        
        if errors:
            print(f"\n   ⚠️ {len(errors)} incorrect predictions (see errors list)")
        
        print("=" * 50)
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_tool": tool_accuracy,
            "errors": errors[:10],  # Return first 10 errors for debugging
        }


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
