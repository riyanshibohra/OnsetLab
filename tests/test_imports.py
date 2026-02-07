"""Test that all public imports work without crashing."""


def test_import_package():
    import onsetlab
    assert hasattr(onsetlab, "__version__")
    assert onsetlab.__version__ == "0.1.0"


def test_import_agent():
    from onsetlab import Agent
    assert Agent is not None


def test_import_mcp():
    from onsetlab import MCPServer
    assert MCPServer is not None


def test_import_router():
    from onsetlab import Router, Strategy
    assert Router is not None
    assert hasattr(Strategy, "REWOO")
    assert hasattr(Strategy, "DIRECT")


def test_import_skills():
    from onsetlab import generate_tool_rules, generate_examples
    assert callable(generate_tool_rules)
    assert callable(generate_examples)


def test_import_benchmark():
    from onsetlab import Benchmark
    assert Benchmark is not None


def test_import_all_tools():
    from onsetlab import (
        BaseTool,
        Calculator,
        DateTime,
        UnitConverter,
        TextProcessor,
        RandomGenerator,
        CodeExecutorTool,
    )
    assert all(
        cls is not None
        for cls in [
            BaseTool,
            Calculator,
            DateTime,
            UnitConverter,
            TextProcessor,
            RandomGenerator,
            CodeExecutorTool,
        ]
    )


def test_import_model_backends():
    from onsetlab.model import BaseModel, OllamaModel
    assert BaseModel is not None
    assert OllamaModel is not None


def test_import_rewoo_components():
    from onsetlab.rewoo.planner import Planner
    from onsetlab.rewoo.executor import Executor
    from onsetlab.rewoo.solver import Solver
    from onsetlab.rewoo.react_fallback import ReactFallback
    from onsetlab.rewoo.verifier import Verifier
    assert all(
        cls is not None
        for cls in [Planner, Executor, Solver, ReactFallback, Verifier]
    )


def test_import_packaging():
    from onsetlab.packaging import export_agent
    assert callable(export_agent)
