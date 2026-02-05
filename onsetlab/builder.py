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

from .utils.schemas import ToolSchema, MCPServerConfig, APIServerConfig, APIToolSchema
from .synthesis.prompts import generate_prompt_for_3b
from .synthesis.data_generator import DataGenerator, DataGenConfig
from .utils.validator import Validator
from .training.unsloth_trainer import UnslothTrainer, TrainerConfig, SUPPORTED_MODELS
from .runtime.packager import AgentPackager, PackageConfig, RuntimeType


@dataclass
class BuildConfig:
    """Configuration for the agent building pipeline."""
    
    # Data generation settings
    num_examples: int = None  # Auto-calculated from tool count if None (25/tool)
    
    # Resume from existing data (skip prompt + data generation)
    # Set to path containing train.jsonl, validation.jsonl, test.jsonl, system_prompt.txt
    existing_data_dir: str = None  # e.g., "./agent_build" to resume from previous run
    
    # Training settings (phi-3.5-fc recommended for best function calling)
    # Set to None for auto-adjustment based on dataset size and tool count
    base_model: str = "phi-3.5-fc"  # Phi-3.5-mini-instruct-hermes-fc (3.8B, best for tools)
    lora_rank: int = None      # Auto: 8 (small) → 16 (medium) → 32 (large)
    lora_alpha: int = None     # Auto: matches rank (or 2x for small datasets)
    epochs: int = None         # Auto: 5 (small) → 3 (medium) → 2 (large)
    learning_rate: float = None  # Auto: 1e-4 (small) → 2e-4 (medium/large)
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
    api_servers: list = None  # NEW: List of APIServerConfig for direct API services
    
    # Paths to generated files
    output_dir: str = None
    model_path: str = None  # Set after training (GGUF or LoRA path)
    package_path: str = None  # Path to package directory
    
    # Build status
    is_trained: bool = False
    is_packaged: bool = False
    
    # Evaluation results (from test.jsonl)
    eval_results: dict = None  # {accuracy, correct, total, per_tool, errors}
    
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
        
        # Find GGUF file - search in multiple locations
        gguf_file = None
        search_paths = []
        
        # Priority 1: model_path directory
        if self.model_path:
            if self.model_path.endswith('.gguf') and os.path.exists(self.model_path):
                gguf_file = self.model_path
            else:
                search_paths.extend([
                    os.path.join(self.model_path, "*.gguf"),
                    os.path.join(self.model_path, "**", "*.gguf"),
                ])
        
        # Priority 2: Standard build output locations
        if self.output_dir:
            search_paths.extend([
                os.path.join(self.output_dir, "model", "gguf", "*.gguf"),
                os.path.join(self.output_dir, "model", "gguf", "**", "*.gguf"),
                os.path.join(self.output_dir, "model", "*.gguf"),
                os.path.join(self.output_dir, "*.gguf"),
            ])
        
        # Priority 3: Colab-specific locations
        search_paths.extend([
            "/content/agent_build/model/gguf/*.gguf",
            "/content/agent_build/model/*.gguf",
            "/content/trained_model/**/*.gguf",
            "/content/*.gguf",
        ])
        
        # Priority 4: Current directory
        search_paths.extend([
            "./*.gguf",
            "./**/*.gguf",
        ])
        
        # Search all paths if not already found
        if not gguf_file:
            all_found = []
            for pattern in search_paths:
                try:
                    files = glob.glob(pattern, recursive=True)
                    files = [f for f in files if f.endswith('.gguf') and os.path.isfile(f)]
                    all_found.extend(files)
                except:
                    continue
            
            # Remove duplicates
            all_found = list(set(all_found))
            
            if all_found:
                # Prefer Q4_K_M quantization, then largest
                q4_files = [f for f in all_found if 'Q4_K_M' in f or 'q4_k_m' in f.lower()]
                if q4_files:
                    gguf_file = max(q4_files, key=os.path.getsize)
                else:
                    gguf_file = max(all_found, key=os.path.getsize)
        
        # Copy GGUF to package
        if gguf_file and os.path.exists(gguf_file):
            dest = os.path.join(final_dir, "model.gguf")
            shutil.copy(gguf_file, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"   ✅ model.gguf ({size_mb:.0f} MB)")
        else:
            print(f"   ⚠️ No GGUF file found - package will need model added manually")
        
        # Copy package files (skip model.gguf if already copied)
        package_dir = self.package_path or os.path.join(self.output_dir, "package")
        if os.path.exists(package_dir):
            for filename in os.listdir(package_dir):
                if filename == "model.gguf":
                    continue  # Already handled above
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
        print(f"🎉 Agent exported! Check your downloads.")
        
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
        mcp_servers: list[MCPServerConfig] = None,
        api_servers: list = None,  # List[APIServerConfig] for direct API services
        api_key: str = None,
        config: BuildConfig = None,
        system_prompt: str = None,  # Pre-generated system prompt (from skill)
        skill: str = None,          # Skill document for guided data generation
    ):
        """
        Initialize the agent builder.
        
        Args:
            problem_statement: Description of what the agent should do
            tools: List of ToolSchema objects (from MCP discovery or API specs)
            mcp_servers: List of MCPServerConfig for MCP-based services
            api_servers: List of APIServerConfig for direct REST API services
            api_key: OpenAI or Anthropic API key for synthesis
            config: Optional BuildConfig for customization
            system_prompt: Pre-generated system prompt (from meta-agent skill generation)
            skill: Skill document for guided data generation (improves quality)
        
        Note:
            You can use mcp_servers, api_servers, or both. At least one must be provided.
        """
        self.problem_statement = problem_statement
        self.tools = tools
        self.mcp_servers = mcp_servers or []
        self.api_servers = api_servers or []
        self.api_key = api_key
        self.config = config or BuildConfig()
        self.provided_system_prompt = system_prompt  # User-provided system prompt
        self.skill = skill  # Skill for guided data generation
        
        # Validate at least one server type is provided
        if not self.mcp_servers and not self.api_servers:
            raise ValueError("At least one of mcp_servers or api_servers must be provided")
        
        # Internal state
        self._system_prompt = None
        self._training_data_path = None
        self._validation_data_path = None  # For stratified validation
        self._test_data_path = None        # For final evaluation
        self._model_path = None
        self._package_path = None
        self._eval_results = None          # Evaluation metrics
        
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
        
        # Check if resuming from existing data
        if self.config.existing_data_dir:
            print(f"📂 Resuming from existing data: {self.config.existing_data_dir}")
            self._load_existing_data()
        else:
            if self.config.num_examples:
                print(f"Target examples: {self.config.num_examples}")
            else:
                config = DataGenConfig()
                auto_examples = config.calculate_total(len(self.tools))
                print(f"Target examples: {auto_examples} (auto-calculated for {len(self.tools)} tools)")
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
            api_servers=self.api_servers,  # NEW: Direct API services
            output_dir=self.config.output_dir,
            model_path=self._model_path,
            package_path=self._package_path,
            is_trained=self._model_path is not None,
            is_packaged=self._package_path is not None,
            eval_results=self._eval_results,  # Accuracy metrics from test.jsonl
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
        if self._eval_results:
            acc = self._eval_results.get("accuracy", 0)
            correct = self._eval_results.get("correct", 0)
            total = self._eval_results.get("total", 0)
            print(f"  📊 Test Accuracy: {acc:.1f}% ({correct}/{total})")
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
        """Step 1: Generate system prompt (and skill if API key available)."""
        print("\n📝 Step 1: Generating system prompt...")
        
        # Use provided system prompt if available
        if self.provided_system_prompt:
            self._system_prompt = self.provided_system_prompt
            print(f"   ✅ Using provided system prompt")
            print(f"   📄 Prompt: {len(self._system_prompt)} chars")
            return
        
        # Try to generate skill-based prompt if API key is available
        if self.api_key and not self.skill:
            try:
                from .synthesis.skill_generator import SkillGenerator
                
                print(f"   🧠 Generating skill for {len(self.tools)} tools...")
                generator = SkillGenerator(api_key=self.api_key)
                
                # Convert tools to dict format if needed
                tools_list = []
                for t in self.tools:
                    if hasattr(t, 'to_dict'):
                        tools_list.append(t.to_dict())
                    elif isinstance(t, dict):
                        tools_list.append(t)
                    else:
                        tools_list.append({
                            'name': getattr(t, 'name', ''),
                            'description': getattr(t, 'description', ''),
                            'parameters': getattr(t, 'parameters', {}),
                            'required_params': getattr(t, 'required_params', [])
                        })
                
                self.skill, self._system_prompt = generator.generate(
                    server_name="Agent",
                    server_description=self.problem_statement,
                    tools=tools_list
                )
                print(f"   ✅ Skill generated ({len(self.skill)} chars)")
                print(f"   📄 System prompt: {len(self._system_prompt)} chars")
                return
                
            except Exception as e:
                print(f"   ⚠️ Skill generation failed: {e}")
                print(f"   📋 Falling back to default prompt")
        
        # Fallback: Use default prompt format
        model_key = self.config.base_model
        model_info = SUPPORTED_MODELS.get(model_key, {})
        tool_format = model_info.get("tool_format", "qwen")
        
        from .synthesis.prompts import get_default_prompt
        self._system_prompt = get_default_prompt(
            problem_statement=self.problem_statement,
            tools=self.tools,
            model_format=tool_format
        )
        
        print(f"   ✅ Using default {tool_format} format")
        print(f"   📄 Prompt: {len(self._system_prompt)} chars")
    
    def _load_existing_data(self):
        """Load existing data from a previous run (skip generation steps)."""
        base_dir = self.config.existing_data_dir
        
        # Load system prompt
        prompt_paths = [
            os.path.join(base_dir, "package", "system_prompt.txt"),
            os.path.join(base_dir, "system_prompt.txt"),
        ]
        for path in prompt_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self._system_prompt = f.read()
                print(f"   ✅ Loaded system prompt ({len(self._system_prompt)} chars)")
                break
        else:
            raise FileNotFoundError(f"No system_prompt.txt found in {base_dir}")
        
        # Load training data paths
        data_dir = os.path.join(base_dir, "data")
        if not os.path.exists(data_dir):
            data_dir = base_dir  # Maybe data is in root
        
        self._training_data_path = os.path.join(data_dir, "train.jsonl")
        self._validation_data_path = os.path.join(data_dir, "validation.jsonl")
        self._test_data_path = os.path.join(data_dir, "test.jsonl")
        
        if not os.path.exists(self._training_data_path):
            raise FileNotFoundError(f"No train.jsonl found in {data_dir}")
        
        # Count examples
        with open(self._training_data_path, 'r') as f:
            self._examples_generated = sum(1 for _ in f)
        
        print(f"   ✅ Loaded {self._examples_generated} training examples")
        print(f"   📄 Train: {self._training_data_path}")
        if os.path.exists(self._validation_data_path):
            print(f"   📄 Validation: {self._validation_data_path}")
        if os.path.exists(self._test_data_path):
            print(f"   📄 Test: {self._test_data_path}")
    
    def _step_2_generate_data(self):
        """Step 2: Generate synthetic training data using single-tool generator."""
        
        # Get model info to determine tool format
        model_key = self.config.base_model
        model_info = SUPPORTED_MODELS.get(model_key, {})
        tool_format = model_info.get("tool_format", "qwen")
        pretrained_on_tools = model_info.get("pretrained_on_tools", False)
        
        # Auto-calculate examples based on tool count if not specified
        gen_config = DataGenConfig(tool_format=tool_format)
        
        num_examples = self.config.num_examples
        if num_examples is None:
            num_examples = gen_config.calculate_total(len(self.tools))
            print(f"\n📊 Step 2: Auto-calculated {num_examples} examples for {len(self.tools)} tools")
            if pretrained_on_tools:
                print(f"   (Model pre-trained on tools - should learn quickly)")
            else:
                print(f"   (25 examples/tool × 75% single-tool + 25% edge cases)")
        else:
            print(f"\n📊 Step 2: Generating {num_examples} training examples...")
        
        print(f"   Tool format: {tool_format}")
        print(f"   Batch size: {gen_config.batch_size} examples/call")
        print(f"   Expected API calls: ~{num_examples // gen_config.batch_size}")
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create data generator
        # CRITICAL: Pass the system prompt so it's included in training examples!
        # Pass skill for guided data generation (improves format correctness)
        generator = DataGenerator(
            tools=self.tools,
            problem_statement=self.problem_statement,
            api_key=self.api_key,
            config=gen_config,
            system_prompt=self._system_prompt,  # Include system prompt in training data!
            skill=self.skill,  # Skill for guided generation (from meta-agent)
        )
        
        if self.skill:
            print(f"   📋 Using skill for guided data generation ({len(self.skill)} chars)")
        
        # Generate and save
        output_dir = os.path.join(self.config.output_dir, "data")
        paths = generator.save(output_dir)
        
        # Use train.jsonl as main training data path
        self._training_data_path = paths.get("train", os.path.join(output_dir, "train.jsonl"))
        
        # Store validation path for stratified split (preserves category balance)
        self._validation_data_path = paths.get("validation", os.path.join(output_dir, "validation.jsonl"))
        
        # Store test path for final evaluation (held-out accuracy measurement)
        self._test_data_path = paths.get("test", os.path.join(output_dir, "test.jsonl"))
        
        # Count total examples generated
        total_generated = generator.stats.get("examples_generated", 0)
        self._examples_generated = total_generated
        
        # Check if we got expected amount (use local num_examples, not config which might be None)
        expected = num_examples
        actual = total_generated
        
        if expected and actual < expected * 0.5:
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
        
        # Show what will be auto-adjusted
        if self.config.lora_rank is None:
            print(f"   LoRA rank: auto (based on dataset size)")
        else:
            print(f"   LoRA rank: {self.config.lora_rank}")
        
        if self.config.epochs is None:
            print(f"   Epochs: auto (based on dataset size)")
        else:
            print(f"   Epochs: {self.config.epochs}")
        
        try:
            # Create trainer config
            # Pass None values to enable auto-adjustment in TrainerConfig
            trainer_config = TrainerConfig(
                base_model=self.config.base_model,
                lora_rank=self.config.lora_rank,      # None = auto-adjust
                lora_alpha=self.config.lora_alpha,    # None = auto-adjust
                epochs=self.config.epochs,             # None = auto-adjust
                learning_rate=self.config.learning_rate,  # None = auto-adjust
                output_dir=os.path.join(self.config.output_dir, "model"),
                save_gguf=self.config.save_gguf,
            )
            
            # Create trainer with validation data path for stratified split
            trainer = UnslothTrainer(
                training_data_path=self._training_data_path,
                validation_data_path=self._validation_data_path,  # Stratified split!
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
            
            # Run training (auto-adjustment happens inside train())
            trainer.train()
            saved_paths = trainer.save()
            
            self._model_path = saved_paths.get("gguf") or saved_paths.get("lora")
            print(f"   ✅ Model saved to: {self._model_path}")
            
            # Run evaluation on held-out test set
            if self._test_data_path and os.path.exists(self._test_data_path):
                print("\n📊 Step 4b: Evaluating on held-out test set...")
                try:
                    self._eval_results = trainer.evaluate(self._test_data_path)
                    if self._eval_results["accuracy"] >= 80:
                        print(f"   ✅ Good accuracy: {self._eval_results['accuracy']:.1f}%")
                    elif self._eval_results["accuracy"] >= 60:
                        print(f"   ⚠️ Moderate accuracy: {self._eval_results['accuracy']:.1f}%")
                        self._warnings.append(f"Model accuracy is {self._eval_results['accuracy']:.1f}% - consider more training data")
                    else:
                        print(f"   ⚠️ Low accuracy: {self._eval_results['accuracy']:.1f}%")
                        self._warnings.append(f"Low accuracy ({self._eval_results['accuracy']:.1f}%) - model may need improvement")
                except Exception as e:
                    print(f"   ⚠️ Evaluation failed: {e}")
                    self._warnings.append(f"Could not evaluate: {e}")
            
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
            
            # Create packager (supports both MCP and API servers)
            packager = AgentPackager(
                agent_name=self.config.agent_name,
                system_prompt=self._system_prompt,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                api_servers=self.api_servers,  # NEW: Direct API services
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
