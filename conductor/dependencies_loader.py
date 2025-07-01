import asyncio
import logging
import sys
import platform
import subprocess
import importlib
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class MinimalDependenciesLoader:
    """Minimal dependencies loader that checks and installs required packages when needed"""

    def __init__(self):
        self.checked_packages = set()
        self.system_info = None


class MinimalDependenciesLoader:
    """Minimal dependencies loader that checks and installs required packages when needed"""

    def __init__(self):
        self.checked_packages = set()
        self.system_info = None

    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements and capabilities"""
        if self.system_info is not None:
            return self.system_info

        try:
            # Basic system info
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'cuda_available': False,
                'cuda_version': None,
                'gpu_count': 0,
                'gpu_memory': [],
                'memory_gb': 16.0  # Default estimate
            }

            # Try to get memory info
            try:
                import psutil
                system_info['memory_gb'] = psutil.virtual_memory().total / (1024 ** 3)
            except ImportError:
                # Rough estimation if psutil not available
                if 'linux' in platform.system().lower():
                    try:
                        with open('/proc/meminfo', 'r') as f:
                            for line in f:
                                if 'MemTotal' in line:
                                    mem_kb = int(line.split()[1])
                                    system_info['memory_gb'] = mem_kb / (1024 * 1024)
                                    break
                    except:
                        pass

            # Check CUDA availability
            try:
                import torch
                if torch.cuda.is_available():
                    system_info['cuda_available'] = True
                    system_info['cuda_version'] = torch.version.cuda
                    system_info['gpu_count'] = torch.cuda.device_count()

                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        system_info['gpu_memory'].append({
                            'device': i,
                            'name': props.name,
                            'memory_gb': props.total_memory / (1024 ** 3)
                        })
            except ImportError:
                logger.info("PyTorch not available - CUDA status unknown")

            self.system_info = system_info
            return system_info

        except Exception as e:
            logger.error(f"Error checking system requirements: {e}")
            return {
                'platform': 'unknown',
                'python_version': platform.python_version(),
                'cpu_count': 1,
                'memory_gb': 8.0,
                'cuda_available': False,
                'cuda_version': None,
                'gpu_count': 0,
                'gpu_memory': []
            }

    def check_package_availability(self, package_name: str) -> bool:
        """Check if a package is available for import"""
        if package_name in self.checked_packages:
            return True

        try:
            importlib.import_module(package_name)
            self.checked_packages.add(package_name)
            logger.debug(f"✓ Package {package_name} is available")
            return True
        except ImportError:
            logger.debug(f"✗ Package {package_name} not available")
            return False

    def get_missing_packages(self, required_packages: List[str]) -> List[str]:
        """Get list of missing packages from required list"""
        missing = []
        for package in required_packages:
            if not self.check_package_availability(package):
                missing.append(package)
        return missing

    async def ensure_dependencies(self, model_name: str, precision: str = "FP16") -> bool:
        """Ensure all dependencies for a model are available"""
        try:
            # Core ML dependencies
            core_deps = ['torch', 'transformers']

            # Additional dependencies based on precision and model type
            additional_deps = []

            if precision in ['4-bit', '8-bit']:
                additional_deps.append('bitsandbytes')

            # Model-specific dependencies
            if 'nllb' in model_name.lower():
                additional_deps.extend(['sentencepiece', 'protobuf'])
            elif 'code' in model_name.lower():
                # Code models might need specific tokenizers
                pass

            all_deps = core_deps + additional_deps
            missing = self.get_missing_packages(all_deps)

            if missing:
                logger.warning(f"Missing packages for {model_name}: {missing}")
                logger.info("Install missing packages with:")

                for package in missing:
                    if package == 'torch':
                        logger.info("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
                    elif package == 'bitsandbytes':
                        logger.info("  pip install bitsandbytes")
                    else:
                        logger.info(f"  pip install {package}")

                return False

            logger.info(f"✓ All dependencies available for {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error checking dependencies for {model_name}: {e}")
            return False

    def install_package(self, package_name: str, extra_index_url: str = None) -> bool:
        """Install a package using pip"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', package_name]

            if extra_index_url:
                cmd.extend(['--index-url', extra_index_url])

            logger.info(f"Installing {package_name}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info(f"✓ Successfully installed {package_name}")
                # Clear checked packages to re-check
                self.checked_packages.discard(package_name)
                return True
            else:
                logger.error(f"Failed to install {package_name}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout installing {package_name}")
            return False
        except Exception as e:
            logger.error(f"Error installing {package_name}: {e}")
            return False

    async def auto_install_dependencies(self, model_name: str, precision: str = "FP16") -> bool:
        """Automatically install missing dependencies"""
        try:
            logger.info(f"Auto-installing dependencies for {model_name} ({precision})")

            # Core dependencies
            if not self.check_package_availability('torch'):
                success = self.install_package(
                    'torch',
                    'https://download.pytorch.org/whl/cu118'
                )
                if not success:
                    return False

            if not self.check_package_availability('transformers'):
                success = self.install_package('transformers')
                if not success:
                    return False

            # Precision-specific dependencies
            if precision in ['4-bit', '8-bit']:
                if not self.check_package_availability('bitsandbytes'):
                    success = self.install_package('bitsandbytes')
                    if not success:
                        logger.warning("Failed to install bitsandbytes - quantization may not work")

            # Model-specific dependencies
            if 'nllb' in model_name.lower():
                for pkg in ['sentencepiece', 'protobuf']:
                    if not self.check_package_availability(pkg):
                        self.install_package(pkg)

            logger.info(f"✓ Dependency installation complete for {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error auto-installing dependencies: {e}")
            return False

    def validate_model_requirements(self, model_name: str, precision: str) -> Dict[str, Any]:
        """Validate if system can handle the model requirements"""
        system_info = self.check_system_requirements()
        validation = {
            'can_load': True,
            'warnings': [],
            'recommendations': []
        }

        try:
            # Estimate memory requirements
            estimated_memory = self._estimate_model_memory(model_name, precision)
            available_memory = system_info.get('memory_gb', 8.0)

            # For CUDA systems, prefer GPU memory if available and sufficient
            if system_info.get('cuda_available'):
                gpu_memories = system_info.get('gpu_memory', [])
                if gpu_memories:
                    max_gpu_memory = max(gpu['memory_gb'] for gpu in gpu_memories)
                    # Use GPU memory if it can fit the model, otherwise use system RAM
                    if estimated_memory <= max_gpu_memory * 0.85:
                        available_memory = max_gpu_memory
                        logger.debug(f"Using GPU memory: {available_memory:.1f}GB")
                    else:
                        logger.debug(
                            f"Model too large for GPU ({max_gpu_memory:.1f}GB), using system RAM: {available_memory:.1f}GB")

            # More generous memory check - allow up to 85% usage
            memory_ratio = estimated_memory / available_memory

            if memory_ratio > 0.85:
                validation['can_load'] = False
                validation['warnings'].append(
                    f"Insufficient memory: need ~{estimated_memory:.1f}GB, available ~{available_memory:.1f}GB ({memory_ratio * 100:.1f}% usage)"
                )

                if precision == 'FP16':
                    validation['recommendations'].append("Try 4-bit quantization to reduce memory usage")
                elif precision == 'FP32':
                    validation['recommendations'].append("Try FP16 or 4-bit quantization")
            elif memory_ratio > 0.75:
                # Warning but still allow
                validation['warnings'].append(
                    f"High memory usage: ~{estimated_memory:.1f}GB of {available_memory:.1f}GB ({memory_ratio * 100:.1f}%)"
                )
                validation['recommendations'].append("Monitor memory usage during operation")

            # Check CUDA for quantization
            if precision in ['4-bit', '8-bit'] and not system_info.get('cuda_available'):
                validation['warnings'].append("Quantization works best with CUDA, may be slow on CPU")

            # CPU cores recommendation
            if system_info.get('cpu_count', 1) < 4:
                validation['warnings'].append("Low CPU core count may affect performance")

            return validation

        except Exception as e:
            logger.error(f"Error validating requirements: {e}")
            validation['can_load'] = False
            validation['warnings'].append(f"Validation error: {e}")
            return validation

    def _estimate_model_memory(self, model_name: str, precision: str) -> float:
        """Estimate memory requirements for a model"""
        # Simple estimation based on model name patterns
        size_estimates = {
            '7b': 7, '8b': 8, '13b': 13, '34b': 34, '70b': 70,
            '3.3b': 3.3, '1.5b': 1.5
        }

        model_lower = model_name.lower()
        base_size = 8  # Default assumption for 8B models

        for size_key, size_gb in size_estimates.items():
            if size_key in model_lower:
                base_size = size_gb
                break

        # Adjust for precision (these are more accurate estimates)
        if precision == 'FP32':
            # 4 bytes per parameter
            return base_size * 4.0
        elif precision == 'FP16':
            # 2 bytes per parameter
            return base_size * 2.0
        elif precision == '4-bit':
            # 0.5 bytes per parameter
            return base_size * 0.5
        elif precision == '8-bit':
            # 1 byte per parameter
            return base_size * 1.0
        else:
            # Default to FP16
            return base_size * 2.0

    def get_installation_instructions(self, missing_packages: List[str]) -> Dict[str, str]:
        """Get installation instructions for missing packages"""
        instructions = {}

        for package in missing_packages:
            if package == 'torch':
                instructions[package] = (
                    "pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8\n"
                    "# Or for CPU only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
                )
            elif package == 'bitsandbytes':
                instructions[package] = (
                    "pip install bitsandbytes  # For quantization support\n"
                    "# Note: Requires CUDA-compatible GPU"
                )
            elif package == 'fastapi':
                instructions[package] = "pip install fastapi uvicorn  # For HTTP server"
            else:
                instructions[package] = f"pip install {package}"

        return instructions

    def get_dependency_report(self) -> Dict[str, Any]:
        """Get comprehensive dependency report"""
        system_info = self.check_system_requirements()

        # Check core packages
        core_packages = ['torch', 'transformers', 'fastapi', 'bitsandbytes']
        package_status = {}

        for package in core_packages:
            available = self.check_package_availability(package)
            package_status[package] = {
                'available': available,
                'required': package in ['torch', 'transformers'],
                'optional': package not in ['torch', 'transformers']
            }

        return {
            'system': system_info,
            'packages': package_status,
            'recommendations': self._get_system_recommendations(system_info)
        }

    def _get_system_recommendations(self, system_info: Dict[str, Any]) -> List[str]:
        """Generate system-specific recommendations"""
        recommendations = []

        if not system_info.get('cuda_available'):
            recommendations.append("Consider installing CUDA for GPU acceleration")

        if system_info.get('memory_gb', 0) < 16:
            recommendations.append("16GB+ RAM recommended for larger models")

        if system_info.get('cpu_count', 0) < 8:
            recommendations.append("8+ CPU cores recommended for optimal performance")

        return recommendations


# Compatibility class for original interface
class DependenciesLoader(MinimalDependenciesLoader):
    """Compatibility wrapper for original DependenciesLoader interface"""

    def __init__(self):
        super().__init__()
        logger.info("DependenciesLoader initialized")

    async def ensure_dependencies(self, model_name: str, precision: str = "FP16") -> bool:
        """Ensure dependencies are available (compatibility method)"""
        return await super().ensure_dependencies(model_name, precision)

    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements (compatibility method)"""
        return super().check_system_requirements()


# For standalone testing
async def main():
    """Test the dependencies loader"""
    loader = DependenciesLoader()

    print("=== System Requirements ===")
    system_info = loader.check_system_requirements()
    for key, value in system_info.items():
        print(f"{key}: {value}")

    print("\n=== Dependency Report ===")
    report = loader.get_dependency_report()

    print("Package Status:")
    for pkg, status in report['packages'].items():
        symbol = "✓" if status['available'] else "✗"
        req_type = "required" if status['required'] else "optional"
        print(f"  {symbol} {pkg} ({req_type})")

    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")

    print("\n=== Testing Model Dependencies ===")
    test_models = [
        ("meta-llama/Llama-3.1-8B-Instruct", "FP16"),
        ("codellama/CodeLlama-7b-hf", "4-bit"),
        ("facebook/nllb-200-3.3B", "FP16")
    ]

    for model, precision in test_models:
        print(f"\nChecking {model} ({precision}):")
        result = await loader.ensure_dependencies(model, precision)
        validation = loader.validate_model_requirements(model, precision)

        print(f"  Dependencies: {'✓' if result else '✗'}")
        print(f"  Can load: {'✓' if validation['can_load'] else '✗'}")

        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")

        if validation['recommendations']:
            for rec in validation['recommendations']:
                print(f"  💡 {rec}")


if __name__ == "__main__":
    asyncio.run(main())