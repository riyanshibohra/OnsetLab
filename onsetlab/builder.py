"""
OnsetLab Agent Builder
======================
Main entry point for building AI agents with fine-tuned SLMs and MCP tools.

Usage:
    from onsetlab import AgentBuilder, ToolSchema, MCPServerConfig
    
    builder = AgentBuilder(
        problem_statement="I need an agent that manages my calendar",
        tools=[ToolSchema(...), ...],
        mcp_servers=[MCPServerConfig(...), ...],
        api_key="sk-..."
    )
    
    agent = builder.build()
    agent.save("./my_agent")
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .utils.schemas import ToolSchema, MCPServerConfig
from .synthesis.prompt_generator import generate_minimal_prompt, PromptGenerator
from .synthesis.data_generator import BatchedDataGenerator, BatchGenConfig
from .utils.validator import Validator
from .training.unsloth_trainer import UnslothTrainer, TrainerConfig, SUPPORTED_MODELS
from .runtime.packager import AgentPackager, PackageConfig, RuntimeType


@dataclass
class BuildConfig:
    """Configuration for the agent building pipeline."""
    
    # Data generation settings (v3 batched generator)
    num_examples: int = 500  # Total examples to generate
    batch_size: int = 10     # Examples per LLM call (10x fewer API calls)
    use_llm_for_prompt: bool = False  # Use LLM for richer system prompt
    
    # Prompt settings
    # "names_only" = Just tool names (shortest, for 20+ tools)
    # "minimal" = Names + description (default, balanced)
    # "full" = Names + description + params (verbose, for <10 tools)
    prompt_detail_level: str = "minimal"
    
    # Training settings (only Qwen2.5-3B supported)
    base_model: str = "qwen2.5-3b"  # Qwen2.5-3B-Instruct
    lora_rank: int = 16
    epochs: int = 3
    skip_training: bool = False  # Set True to skip training (synthesis only)
    
    # Output settings
    output_dir: str = "./agent_build"
    agent_name: str = "my_agent"
    save_gguf: bool = True  # Save GGUF for Ollama
    runtime: str = "both"  # "ollama", "python", or "both"


@dataclass
class Agent:
    """
    Represents a built agent.
    
    Contains all the generated artifacts and provides methods
    for testing and saving the agent.
    """
    name: str
    problem_statement: str
    system_prompt: str
    training_data_path: str
    tools: list[ToolSchema]
    mcp_servers: list[MCPServerConfig]
    
    # Paths to generated files
    output_dir: str = None
    model_path: str = None  # Set after training (GGUF or LoRA path)
    package_path: str = None  # Path to package directory
    
    # Build status
    is_trained: bool = False
    is_packaged: bool = False
    
    def export(self, output_path: str = None) -> str:
        """
        Export a complete, downloadable agent package with GGUF model.
        
        This combines the trained model (GGUF) with the package files
        into a single zip file ready for distribution.
        
        Args:
            output_path: Path for the zip file (default: ./agent_name.zip)
            
        Returns:
            Path to the created zip file
        """
        import shutil
        import glob
        
        if not self.is_trained:
            raise ValueError("Agent must be trained before exporting. Run build() first.")
        
        # Determine output paths
        zip_name = output_path or f"./{self.name}.zip"
        if zip_name.endswith('.zip'):
            zip_name = zip_name[:-4]  # shutil.make_archive adds .zip
        
        final_dir = f"{zip_name}_temp"
        os.makedirs(final_dir, exist_ok=True)
        
        print(f"📦 Exporting agent package...")
        
        # Find and copy GGUF file
        gguf_file = None
        if self.model_path:
            # Search for GGUF in model path
            search_paths = [
                os.path.join(self.model_path, "*.gguf"),
                os.path.join(self.model_path, "**", "*.gguf"),
                os.path.join(self.output_dir, "model", "gguf", "*.gguf"),
                os.path.join(self.output_dir, "model", "gguf", "**", "*.gguf"),
            ]
            
            for pattern in search_paths:
                files = glob.glob(pattern, recursive=True)
                if files:
                    # Prefer Q4_K_M quantization
                    gguf_file = next(
                        (f for f in files if 'Q4_K_M' in f or 'q4_k_m' in f.lower()),
                        files[0]
                    )
                    break
        
        if gguf_file and os.path.exists(gguf_file):
            shutil.copy(gguf_file, os.path.join(final_dir, "model.gguf"))
            size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
            print(f"   ✅ model.gguf ({size_mb:.0f} MB)")
        else:
            print(f"   ⚠️ No GGUF file found - package will need model added manually")
        
        # Copy package files
        package_dir = self.package_path or os.path.join(self.output_dir, "package")
        if os.path.exists(package_dir):
            for filename in os.listdir(package_dir):
                src = os.path.join(package_dir, filename)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(final_dir, filename))
                    print(f"   ✅ {filename}")
        
        # Create zip
        shutil.make_archive(zip_name, 'zip', final_dir)
        
        # Cleanup temp directory
        shutil.rmtree(final_dir)
        
        zip_path = f"{zip_name}.zip"
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"\n✅ Package exported: {zip_path} ({size_mb:.0f} MB)")
        
        return zip_path
    
    def test(self, query: str) -> str:
        """
        Test the agent with a query.
        
        Note: Full testing requires the model to be trained.
        For now, this shows what tool would be called.
        """
        if not self.is_trained:
            return (
                f"⚠️ Agent not yet trained. After training, this query would be processed.\n"
                f"Query: {query}\n"
                f"Available tools: {', '.join(t.name for t in self.tools)}"
            )
        
        # TODO: Implement actual inference after training module is built
        return f"[Inference not implemented yet] Query: {query}"
    
    def save(self, path: str = None) -> str:
        """
        Save the agent to a directory.
        
        Args:
            path: Output directory (defaults to output_dir/agent_name)
            
        Returns:
            Path to saved agent directory
        """
        save_path = path or os.path.join(self.output_dir, self.name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save system prompt
        with open(os.path.join(save_path, "system_prompt.txt"), "w") as f:
            f.write(self.system_prompt)
        
        # Copy training data (if not already in save location)
        if self.training_data_path and os.path.exists(self.training_data_path):
            dest_path = os.path.join(save_path, "training_data.jsonl")
            src_abs = os.path.abspath(self.training_data_path)
            dest_abs = os.path.abspath(dest_path)
            if src_abs != dest_abs:
                import shutil
                shutil.copy(self.training_data_path, dest_path)
        
        # Save config
        config = {
            "name": self.name,
            "problem_statement": self.problem_statement,
            "tools": [t.to_dict() for t in self.tools],
            "mcp_servers": [s.to_dict() for s in self.mcp_servers],
            "is_trained": self.is_trained,
            "is_packaged": self.is_packaged,
        }
        with open(os.path.join(save_path, "agent_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Agent saved to: {save_path}")
        return save_path
    
    def __repr__(self) -> str:
        status = []
        if self.is_trained:
            status.append("trained")
        if self.is_packaged:
            status.append("packaged")
        status_str = ", ".join(status) if status else "synthesis complete"
        return f"Agent(name='{self.name}', status='{status_str}')"


class AgentBuilder:
    """
    Main entry point for building AI agents.
    
    Orchestrates the complete pipeline:
    1. System prompt generation
    2. Training data synthesis
    3. Model fine-tuning
    4. Agent packaging
    
    Example:
        >>> from onsetlab import AgentBuilder, ToolSchema, MCPServerConfig
        >>> 
        >>> builder = AgentBuilder(
        ...     problem_statement="I need an agent that manages my calendar",
        ...     tools=[
        ...         ToolSchema(name="list-events", description="...", parameters={...}),
        ...     ],
        ...     mcp_servers=[
        ...         MCPServerConfig(package="@cocal/google-calendar-mcp", auth_type="oauth"),
        ...     ],
        ...     api_key="sk-..."
        ... )
        >>> 
        >>> agent = builder.build()
        >>> agent.save("./my_agent")
    """
    
    def __init__(
        self,
        problem_statement: str,
        tools: list[ToolSchema],
        mcp_servers: list[MCPServerConfig],
        api_key: str,
        config: BuildConfig = None,
    ):
        """
        Initialize the agent builder.
        
        Args:
            problem_statement: Description of what the agent should do
            tools: List of ToolSchema objects (from MCP discovery)
            mcp_servers: List of MCPServerConfig objects
            api_key: OpenAI or Anthropic API key for synthesis
            config: Optional BuildConfig for customization
        """
        self.problem_statement = problem_statement
        self.tools = tools
        self.mcp_servers = mcp_servers
        self.api_key = api_key
        self.config = config or BuildConfig()
        
        # Internal state
        self._system_prompt = None
        self._training_data_path = None
        self._model_path = None
        self._package_path = None
        
        # Build status tracking
        self._errors = []
        self._warnings = []
        self._examples_generated = 0
    
    def build(self) -> Agent:
        """
        Run the complete agent building pipeline.
        
        Returns:
            Agent object with all generated artifacts
        """
        print("=" * 60)
        print("🚀 OnsetLab Agent Builder")
        print("=" * 60)
        print(f"\nProblem: {self.problem_statement}")
        print(f"Tools: {len(self.tools)}")
        print(f"MCP Servers: {len(self.mcp_servers)}")
        print(f"Target examples: {self.config.num_examples}")
        
        # Step 1: Generate system prompt
        self._step_1_generate_prompt()
        
        # Step 2: Generate training data
        self._step_2_generate_data()
        
        # Step 3: Validate training data
        self._step_3_validate_data()
        
        # Step 4: Fine-tune model (TODO)
        self._step_4_train_model()
        
        # Step 5: Package agent (TODO)
        self._step_5_package_agent()
        
        # Create and return Agent object
        agent = Agent(
            name=self.config.agent_name,
            problem_statement=self.problem_statement,
            system_prompt=self._system_prompt,
            training_data_path=self._training_data_path,
            tools=self.tools,
            mcp_servers=self.mcp_servers,
            output_dir=self.config.output_dir,
            model_path=self._model_path,
            package_path=self._package_path,
            is_trained=self._model_path is not None,
            is_packaged=self._package_path is not None,
        )
        
        # Print final status
        print("\n" + "=" * 60)
        
        if self._errors:
            print("❌ Build completed with ERRORS!")
            print("=" * 60)
            for error in self._errors:
                print(f"  ❌ {error}")
        elif self._warnings:
            print("⚠️ Build completed with WARNINGS!")
            print("=" * 60)
            for warning in self._warnings:
                print(f"  ⚠️ {warning}")
        else:
            print("✅ Build complete!")
            print("=" * 60)
        
        print(f"\nGenerated artifacts:")
        print(f"  📄 System prompt: {len(self._system_prompt)} chars")
        print(f"  📄 Training data: {self._training_data_path}")
        print(f"  📊 Examples: {self._examples_generated}")
        if self._model_path:
            print(f"  🧠 Model: {self._model_path}")
        if self._package_path:
            print(f"  📦 Package: {self._package_path}")
        
        # Check if we have enough examples
        min_examples = 50
        if self._examples_generated < min_examples:
            print(f"\n❌ CRITICAL: Only {self._examples_generated} examples generated!")
            print(f"   Minimum recommended: {min_examples}")
            print(f"   This is likely due to LLM response parsing failures.")
            print(f"   Try running again or reducing num_examples.")
        
        # Show next steps
        if self._package_path:
            print(f"\n🚀 Next steps:")
            print(f"   cd {self._package_path}")
            if self.config.runtime in ("ollama", "both"):
                print(f"   ollama create {self.config.agent_name} -f Modelfile")
                print(f"   ollama run {self.config.agent_name}")
            if self.config.runtime in ("python", "both"):
                print(f"   python agent.py --interactive")
        
        return agent
    
    def _step_1_generate_prompt(self):
        """Step 1: Generate system prompt."""
        print("\n📝 Step 1: Generating system prompt...")
        print(f"   Detail level: {self.config.prompt_detail_level}")
        
        if self.config.use_llm_for_prompt:
            generator = PromptGenerator(api_key=self.api_key)
            self._system_prompt = generator.generate(
                self.problem_statement,
                self.tools,
                use_llm=True
            )
            print(f"   ✅ Generated LLM-based prompt ({len(self._system_prompt)} chars)")
        else:
            self._system_prompt = generate_minimal_prompt(
                self.problem_statement,
                self.tools,
                detail_level=self.config.prompt_detail_level
            )
            print(f"   ✅ Generated template-based prompt ({len(self._system_prompt)} chars)")
    
    def _step_2_generate_data(self):
        """Step 2: Generate synthetic training data using batched v3 generator."""
        print(f"\n📊 Step 2: Generating {self.config.num_examples} training examples (batched)...")
        print(f"   Batch size: {self.config.batch_size} examples/call")
        print(f"   Expected API calls: ~{self.config.num_examples // self.config.batch_size}")
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create v3 batched generator config
        gen_config = BatchGenConfig(
            total_examples=self.config.num_examples,
            batch_size=self.config.batch_size,
        )
        
        # Create v3 generator (flat tool list, no server context)
        # CRITICAL: Pass the system prompt so it's included in training examples!
        generator = BatchedDataGenerator(
            tools=self.tools,
            problem_statement=self.problem_statement,
            api_key=self.api_key,
            config=gen_config,
            system_prompt=self._system_prompt,  # Include system prompt in training data!
        )
        
        # Generate and save
        output_dir = os.path.join(self.config.output_dir, "data")
        paths = generator.save(output_dir)
        
        # Use train.jsonl as main training data path
        self._training_data_path = paths.get("train", os.path.join(output_dir, "train.jsonl"))
        
        # Count total examples generated
        total_generated = generator.stats.get("examples_generated", 0)
        self._examples_generated = total_generated
        
        # Check if we got expected amount
        expected = self.config.num_examples
        actual = total_generated
        
        if actual < expected * 0.5:
            self._warnings.append(f"Only {actual}/{expected} examples generated (target missed by >50%)")
            print(f"   ⚠️ Only generated {actual}/{expected} examples")
        else:
            print(f"   ✅ Generated {actual} examples")
    
    def _step_3_validate_data(self):
        """Step 3: Validate generated training data."""
        print("\n✓ Step 3: Validating training data...")
        
        validator = Validator(tools=self.tools)
        result = validator.validate(self._training_data_path)
        
        print(f"   Total: {result.total_examples}")
        print(f"   Valid: {result.valid_examples} ✅")
        print(f"   Invalid: {result.invalid_examples} ❌")
        print(f"   Quality: {result.quality_score:.1f}%")
        
        # Show error breakdown if there are errors
        if result.error_counts:
            print(f"\n   📋 Error breakdown:")
            for error_type, count in sorted(result.error_counts.items(), key=lambda x: -x[1]):
                print(f"      - {error_type}: {count}")
            
            # Show sample errors (first 3)
            if result.errors:
                print(f"\n   📋 Sample errors (first 3):")
                for error in result.errors[:3]:
                    print(f"      Line {error.line_number}: {error.message[:80]}")
        
        if result.quality_score < 80:
            self._warnings.append(f"Low quality: {result.quality_score:.1f}%")
            print("\n   ⚠️ Quality is low - consider regenerating")
        elif result.quality_score >= 90:
            print("\n   ✅ Good quality - ready for training!")
        
        if result.total_examples == 0:
            self._errors.append("No training examples generated!")
            print("   ❌ No training examples generated!")
    
    def _step_4_train_model(self):
        """Step 4: Fine-tune the model with Unsloth."""
        print("\n🎯 Step 4: Fine-tuning model...")
        
        if self.config.skip_training:
            print("   ⏭️ Skipping training (skip_training=True)")
            return
        
        # Get model info
        model_key = self.config.base_model
        if model_key in SUPPORTED_MODELS:
            model_info = SUPPORTED_MODELS[model_key]
            print(f"   Model: {model_info['name']} ({model_info['size']})")
        else:
            print(f"   Model: {model_key}")
        
        print(f"   LoRA rank: {self.config.lora_rank}")
        print(f"   Epochs: {self.config.epochs}")
        
        try:
            # Create trainer config
            trainer_config = TrainerConfig(
                base_model=self.config.base_model,
                lora_rank=self.config.lora_rank,
                epochs=self.config.epochs,
                output_dir=os.path.join(self.config.output_dir, "model"),
                save_gguf=self.config.save_gguf,
            )
            
            # Create trainer
            trainer = UnslothTrainer(
                training_data_path=self._training_data_path,
                system_prompt=self._system_prompt,
                tools=self.tools,
                config=trainer_config,
            )
            
            # Check if GPU available
            if not trainer.has_gpu:
                print("\n   ⚠️ No GPU detected - cannot train locally")
                print("   📋 To train, run this in Google Colab with T4 GPU")
                self._warnings.append("No GPU - training skipped")
                return
            
            # Run training
            trainer.train()
            saved_paths = trainer.save()
            
            self._model_path = saved_paths.get("gguf") or saved_paths.get("lora")
            print(f"   ✅ Model saved to: {self._model_path}")
            
        except ImportError as e:
            print(f"\n   ⚠️ Missing dependencies for training: {e}")
            print("   Install with: pip install unsloth transformers datasets trl")
            self._warnings.append("Training dependencies not installed")
        except Exception as e:
            print(f"\n   ❌ Training failed: {e}")
            self._errors.append(f"Training failed: {e}")
    
    def _step_5_package_agent(self):
        """Step 5: Package the agent for deployment."""
        print("\n📦 Step 5: Packaging agent...")
        
        # Map string runtime to enum
        runtime_map = {
            "ollama": RuntimeType.OLLAMA,
            "python": RuntimeType.PYTHON,
            "both": RuntimeType.BOTH,
        }
        runtime = runtime_map.get(self.config.runtime, RuntimeType.BOTH)
        
        print(f"   Runtime: {runtime.value}")
        
        try:
            # Create packager config
            package_config = PackageConfig(
                runtime=runtime,
                agent_name=self.config.agent_name,
                output_dir=os.path.join(self.config.output_dir, "package"),
                include_readme=True,
            )
            
            # Create packager
            packager = AgentPackager(
                agent_name=self.config.agent_name,
                system_prompt=self._system_prompt,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                model_path=self._model_path,
                config=package_config,
            )
            
            # Package the agent
            package_path = packager.package()
            self._package_path = package_path
            
        except Exception as e:
            print(f"   ❌ Packaging failed: {e}")
            self._errors.append(f"Packaging failed: {e}")
            self._package_path = None
