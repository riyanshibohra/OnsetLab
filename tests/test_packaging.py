#!/usr/bin/env python3
"""
Test OnsetLab Packaging - End-to-End

Tests all export formats:
1. Config (YAML/JSON)
2. Docker
3. Binary/Script
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime, UnitConverter, TextProcessor
from .conftest import requires_ollama

pytestmark = requires_ollama


def test_config_export():
    """Test config export (YAML and JSON)."""
    print("\n" + "=" * 60)
    print("TEST: Config Export")
    print("=" * 60)
    
    # Create agent with tools
    agent = Agent(
        model="phi3.5",
        tools=[Calculator(), DateTime(), UnitConverter()],
        memory=True,
        verify=True,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test YAML export
        yaml_path = os.path.join(tmpdir, "agent.yaml")
        result = agent.export("config", yaml_path)
        
        assert os.path.exists(result), f"YAML file not created: {result}"
        with open(result) as f:
            content = f.read()
        assert "model: phi3.5" in content, "Model not in YAML"
        assert "Calculator" in content, "Calculator not in YAML"
        print(f"  [PASS] YAML export: {result}")
        
        # Test JSON export
        json_path = os.path.join(tmpdir, "agent.json")
        result = agent.export("config", json_path)
        
        assert os.path.exists(result), f"JSON file not created: {result}"
        with open(result) as f:
            content = f.read()
        assert '"model": "phi3.5"' in content, "Model not in JSON"
        print(f"  [PASS] JSON export: {result}")
    
    print("\n  CONFIG EXPORT: ALL TESTS PASSED")


def test_docker_export():
    """Test Docker export."""
    print("\n" + "=" * 60)
    print("TEST: Docker Export")
    print("=" * 60)
    
    agent = Agent(
        model="phi3.5",
        tools=[Calculator(), TextProcessor()],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test basic Docker export
        docker_dir = os.path.join(tmpdir, "docker_agent")
        result = agent.export("docker", docker_dir)
        
        # Check all files created
        expected_files = [
            "Dockerfile",
            "docker-compose.yml", 
            "agent_config.yaml",
            "entrypoint.py",
            "requirements.txt",
            "README.md",
        ]
        
        for filename in expected_files:
            filepath = os.path.join(result, filename)
            assert os.path.exists(filepath), f"Missing: {filename}"
            print(f"  [PASS] Created: {filename}")
        
        # Verify Dockerfile content
        with open(os.path.join(result, "Dockerfile")) as f:
            dockerfile = f.read()
        assert "python:3.11-slim" in dockerfile, "Base image missing"
        assert "EXPOSE 8000" in dockerfile, "Port not exposed"
        
        # Verify docker-compose.yml
        with open(os.path.join(result, "docker-compose.yml")) as f:
            compose = f.read()
        assert "ollama/ollama:latest" in compose, "Ollama service missing"
        assert "depends_on" in compose, "Dependencies missing"
        
        print("\n  Testing Docker with bundled Ollama...")
        
        # Test with bundled Ollama
        docker_dir2 = os.path.join(tmpdir, "docker_bundled")
        result2 = agent.export("docker", docker_dir2, include_ollama=True)
        
        with open(os.path.join(result2, "Dockerfile")) as f:
            dockerfile = f.read()
        assert "FROM ollama/ollama" in dockerfile, "Ollama base missing"
        assert "start.sh" in dockerfile, "Start script missing"
        print(f"  [PASS] Bundled Ollama export")
        
        assert os.path.exists(os.path.join(result2, "start.sh")), "start.sh missing"
        print(f"  [PASS] Start script created")
    
    print("\n  DOCKER EXPORT: ALL TESTS PASSED")


def test_binary_export():
    """Test binary/script export."""
    print("\n" + "=" * 60)
    print("TEST: Binary/Script Export")
    print("=" * 60)
    
    agent = Agent(
        model="phi3.5",
        tools=[Calculator(), DateTime()],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test script export
        script_path = os.path.join(tmpdir, "my_agent.py")
        result = agent.export("binary", script_path)
        
        assert os.path.exists(result), f"Script not created: {result}"
        assert result.endswith(".py"), "Wrong extension"
        print(f"  [PASS] Script created: {result}")
        
        # Verify script content
        with open(result) as f:
            content = f.read()
        
        assert "CONFIG =" in content, "Embedded config missing"
        assert "phi3.5" in content, "Model not embedded"
        assert "Calculator" in content, "Tools not embedded"
        assert "def create_agent():" in content, "create_agent missing"
        assert "def main():" in content, "main missing"
        assert "--serve" in content, "Server mode missing"
        print(f"  [PASS] Script content verified")
        
        # Check executable permission (Unix only)
        if sys.platform != "win32":
            import stat
            mode = os.stat(result).st_mode
            is_executable = mode & stat.S_IXUSR
            assert is_executable, "Script not executable"
            print(f"  [PASS] Script is executable")
    
    print("\n  BINARY EXPORT: ALL TESTS PASSED")


def test_sdk_export_method():
    """Test export via Agent.export() method."""
    print("\n" + "=" * 60)
    print("TEST: SDK Agent.export() Method")
    print("=" * 60)
    
    # Create agent
    agent = Agent(
        model="qwen2.5:3b",
        tools=[Calculator(), DateTime(), UnitConverter(), TextProcessor()],
        memory=True,
        verify=True,
        routing=True,
    )
    
    print(f"  Created agent: {agent}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test all formats via SDK method
        formats = [
            ("config", "config.yaml", {}),
            ("docker", "docker_out", {}),
            ("docker", "docker_bundled", {"include_ollama": True}),
            ("binary", "standalone.py", {}),
        ]
        
        for fmt, output, kwargs in formats:
            output_path = os.path.join(tmpdir, output)
            result = agent.export(fmt, output_path, **kwargs)
            
            assert os.path.exists(result), f"Export failed: {fmt} -> {result}"
            
            extra_info = f" (include_ollama)" if kwargs.get("include_ollama") else ""
            print(f"  [PASS] agent.export('{fmt}'{extra_info}) -> {Path(result).name}")
    
    print("\n  SDK EXPORT METHOD: ALL TESTS PASSED")


def main():
    print("=" * 60)
    print("OnsetLab Packaging - End-to-End Tests")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    tests = [
        ("Config Export", test_config_export),
        ("Docker Export", test_docker_export),
        ("Binary Export", test_binary_export),
        ("SDK Export Method", test_sdk_export_method),
    ]
    
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, status in results:
        icon = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {name}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
