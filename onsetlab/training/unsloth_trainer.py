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
# We use Qwen2.5-3B-Instruct as the ONLY supported model.
# Reasons:
#   - Best tool calling accuracy among non-gated SLMs
#   - 3B parameters = good balance of quality vs. speed
#   - 32K context window = handles complex prompts
#   - Non-gated = no Hugging Face auth required
#
MODEL_CONFIG = {
    "name": "Qwen2.5-3B-Instruct",
    "unsloth_id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "size": "3B",
    "context_length": 32768,
}

# Keep for backwards compatibility
SUPPORTED_MODELS = {
    "qwen2.5-3b": MODEL_CONFIG,
}

DEFAULT_MODEL = "qwen2.5-3b"


@dataclass
class TrainerConfig:
    """Configuration for fine-tuning with Qwen2.5-3B-Instruct."""
    
    # Model settings (only Qwen2.5-3B is supported)
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
            # Medium dataset: balanced
            defaults = {
                "lora_rank": 16,
                "lora_alpha": 16,
                "epochs": 3,
                "learning_rate": 2e-4,
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
        Load the base model with Unsloth.
        
        Args:
            num_examples: If provided, auto-adjusts LoRA params based on dataset size
            num_tools: If provided, adjusts rank based on tool count
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
        
        # Ensure LoRA config is set (should already be done by train())
        # This handles the case where load_model() is called directly
        if self.config.lora_rank is None or self.config.lora_alpha is None:
            if num_examples:
                self.config.auto_adjust_for_dataset(num_examples, num_tools)
            else:
                # Use safe defaults if not auto-adjusted
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
        """Run the fine-tuning process using Unsloth's recommended pattern."""
        # Disable Weights & Biases prompting
        import os
        os.environ["WANDB_DISABLED"] = "true"
        
        from unsloth import is_bfloat16_supported
        from datasets import Dataset
        
        # Handle different trl versions - try newer API first
        try:
            from trl import SFTTrainer, SFTConfig
            use_sft_config = True
        except ImportError:
            from trl import SFTTrainer
            from transformers import TrainingArguments
            use_sft_config = False
            print("   ⚠️ Using older trl API (TrainingArguments)")
        
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
        
        # Unsloth's recommended trainer setup
        print("🚀 Starting training (with validation)...")
        
        if use_sft_config:
            # New trl API (v0.8+): Use SFTConfig
            sft_config = SFTConfig(
                # Training settings
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
                eval_strategy="epoch",
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                optim="adamw_8bit",
                seed=42,
                report_to="none",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                # SFT-specific settings
                max_seq_length=self.config.max_seq_length,
                dataset_text_field="text",  # Use column name, more reliable
                dataset_num_proc=2,
                packing=False,
            )
            
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,  # Works in both old and new trl
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                args=sft_config,
            )
        else:
            # Old trl API: Use TrainingArguments + direct params
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
                evaluation_strategy="epoch",  # Old API uses evaluation_strategy
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                optim="adamw_8bit",
                seed=42,
                report_to="none",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
            
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                dataset_num_proc=2,
                packing=False,
                args=training_args,
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
