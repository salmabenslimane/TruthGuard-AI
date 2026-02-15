#!/usr/bin/env python3
"""
Benchmark script to compare training performance: Local vs Docker
This script measures training time, memory usage, and model performance
"""

import time
import json
import subprocess
import psutil
import os
from datetime import datetime
from pathlib import Path

class PerformanceBenchmark:
    """Benchmark class to compare local and Docker training"""
    
    def __init__(self):
        self.results = {
            'local': {},
            'docker': {},
            'comparison': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def get_system_info(self):
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_physical': psutil.cpu_count(logical=False),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': subprocess.check_output(['python', '--version']).decode().strip()
        }
    
    def measure_local_training(self, num_epochs=1):
        """
        Measure local training performance
        Note: Uses reduced epochs for faster benchmarking
        """
        print("="*70)
        print("BENCHMARK 1: LOCAL EXECUTION (without Docker)")
        print("="*70)
        
        # Temporarily modify config for faster benchmark
        config_backup = None
        config_path = Path("src/config.py")
        
        if config_path.exists():
            config_backup = config_path.read_text()
            modified_config = config_backup.replace("NUM_EPOCHS = 3", f"NUM_EPOCHS = {num_epochs}")
            config_path.write_text(modified_config)
        
        # Track memory before training
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)  # MB
        
        # Measure training time
        start_time = time.time()
        
        try:
            # Run training locally
            result = subprocess.run(
                ['python', 'src/train.py'],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Track memory after training
            mem_after = process.memory_info().rss / (1024**2)  # MB
            mem_used = mem_after - mem_before
            
            # Extract metrics from logs
            metrics = self._extract_metrics_from_logs()
            
            self.results['local'] = {
                'status': 'success' if result.returncode == 0 else 'failed',
                'training_time_seconds': round(training_time, 2),
                'training_time_minutes': round(training_time / 60, 2),
                'memory_used_mb': round(mem_used, 2),
                'peak_memory_mb': round(mem_after, 2),
                'metrics': metrics,
                'stdout_length': len(result.stdout),
                'stderr_length': len(result.stderr)
            }
            
            print(f"\n✓ Local training completed successfully!")
            print(f"  Time: {self.results['local']['training_time_minutes']:.2f} minutes")
            print(f"  Memory: {self.results['local']['memory_used_mb']:.2f} MB")
            
        except subprocess.TimeoutExpired:
            self.results['local'] = {
                'status': 'timeout',
                'error': 'Training exceeded 30 minutes timeout'
            }
            print("\n✗ Local training timed out!")
            
        except Exception as e:
            self.results['local'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"\n✗ Local training error: {e}")
        
        finally:
            # Restore original config
            if config_backup and config_path.exists():
                config_path.write_text(config_backup)
    
    def measure_docker_training(self, num_epochs=1):
        """
        Measure Docker training performance
        """
        print("\n" + "="*70)
        print("BENCHMARK 2: DOCKER EXECUTION")
        print("="*70)
        
        # Temporarily modify config for faster benchmark
        config_backup = None
        config_path = Path("src/config.py")
        
        if config_path.exists():
            config_backup = config_path.read_text()
            modified_config = config_backup.replace("NUM_EPOCHS = 3", f"NUM_EPOCHS = {num_epochs}")
            config_path.write_text(modified_config)
        
        start_time = time.time()
        
        try:
            # Build Docker image
            print("\nBuilding Docker image...")
            build_start = time.time()
            build_result = subprocess.run(
                ['docker', 'build', '-t', 'truthguard-training', '-f', 'Dockerfile', '.'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for build
            )
            build_time = time.time() - build_start
            
            if build_result.returncode != 0:
                raise Exception(f"Docker build failed: {build_result.stderr}")
            
            print(f"✓ Docker image built in {build_time:.2f} seconds")
            
            # Run training in Docker
            print("\nRunning training in Docker...")
            train_start = time.time()
            train_result = subprocess.run(
                [
                    'docker', 'run', '--rm',
                    '-v', f'{os.getcwd()}/models:/app/models',
                    '-v', f'{os.getcwd()}/logs:/app/logs',
                    '-v', f'{os.getcwd()}/data:/app/data',
                    'truthguard-training'
                ],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            train_time = time.time() - train_start
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Get Docker image size
            image_size_result = subprocess.run(
                ['docker', 'images', 'truthguard-training', '--format', '{{.Size}}'],
                capture_output=True,
                text=True
            )
            image_size = image_size_result.stdout.strip()
            
            # Extract metrics from logs
            metrics = self._extract_metrics_from_logs()
            
            self.results['docker'] = {
                'status': 'success' if train_result.returncode == 0 else 'failed',
                'total_time_seconds': round(total_time, 2),
                'total_time_minutes': round(total_time / 60, 2),
                'build_time_seconds': round(build_time, 2),
                'training_time_seconds': round(train_time, 2),
                'training_time_minutes': round(train_time / 60, 2),
                'image_size': image_size,
                'metrics': metrics,
                'stdout_length': len(train_result.stdout),
                'stderr_length': len(train_result.stderr)
            }
            
            print(f"\n✓ Docker training completed successfully!")
            print(f"  Build time: {self.results['docker']['build_time_seconds']:.2f} seconds")
            print(f"  Training time: {self.results['docker']['training_time_minutes']:.2f} minutes")
            print(f"  Total time: {self.results['docker']['total_time_minutes']:.2f} minutes")
            print(f"  Image size: {image_size}")
            
        except subprocess.TimeoutExpired:
            self.results['docker'] = {
                'status': 'timeout',
                'error': 'Docker training exceeded timeout'
            }
            print("\n✗ Docker training timed out!")
            
        except Exception as e:
            self.results['docker'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"\n✗ Docker training error: {e}")
        
        finally:
            # Restore original config
            if config_backup and config_path.exists():
                config_path.write_text(config_backup)
            
            # Cleanup Docker container and image
            subprocess.run(['docker', 'rmi', 'truthguard-training'], 
                         capture_output=True, stderr=subprocess.DEVNULL)
    
    def _extract_metrics_from_logs(self):
        """Extract training metrics from log files"""
        log_file = Path("logs/training_results.json")
        
        if not log_file.exists():
            return None
        
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            return {
                'test_accuracy': data.get('test_metrics', {}).get('accuracy'),
                'test_f1_weighted': data.get('test_metrics', {}).get('f1_weighted'),
                'best_val_accuracy': data.get('best_val_accuracy'),
                'training_time': data.get('training_time')
            }
        except:
            return None
    
    def compare_results(self):
        """Compare local vs Docker results"""
        print("\n" + "="*70)
        print("COMPARISON: LOCAL vs DOCKER")
        print("="*70)
        
        if self.results['local'].get('status') != 'success' or \
           self.results['docker'].get('status') != 'success':
            print("\n⚠ Cannot compare - one or both benchmarks failed")
            return
        
        local_time = self.results['local']['training_time_seconds']
        docker_time = self.results['docker']['training_time_seconds']
        
        overhead = ((docker_time - local_time) / local_time) * 100
        speedup = local_time / docker_time
        
        self.results['comparison'] = {
            'docker_overhead_percent': round(overhead, 2),
            'speedup_factor': round(speedup, 3),
            'time_difference_seconds': round(docker_time - local_time, 2),
            'local_faster': local_time < docker_time
        }
        
        print(f"\nTraining Time:")
        print(f"  Local:  {self.results['local']['training_time_minutes']:.2f} minutes")
        print(f"  Docker: {self.results['docker']['training_time_minutes']:.2f} minutes")
        print(f"  Docker overhead: {overhead:+.2f}%")
        
        if overhead < 10:
            print(f"\n✓ Docker overhead is minimal (<10%), excellent for production use!")
        elif overhead < 20:
            print(f"\n✓ Docker overhead is acceptable (<20%), good for production use")
        else:
            print(f"\n⚠ Docker overhead is significant (>20%), consider optimization")
        
        # Compare metrics if available
        local_metrics = self.results['local'].get('metrics')
        docker_metrics = self.results['docker'].get('metrics')
        
        if local_metrics and docker_metrics:
            print(f"\nModel Performance:")
            print(f"  Local accuracy:  {local_metrics.get('test_accuracy', 'N/A')}")
            print(f"  Docker accuracy: {docker_metrics.get('test_accuracy', 'N/A')}")
            
            if local_metrics.get('test_accuracy') == docker_metrics.get('test_accuracy'):
                print(f"  ✓ Identical results - perfect reproducibility!")
    
    def save_results(self, output_file='logs/benchmark_results.json'):
        """Save benchmark results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # Add system info
        self.results['system_info'] = self.get_system_info()
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        print("\nSystem Information:")
        sys_info = self.results.get('system_info', {})
        for key, value in sys_info.items():
            print(f"  {key}: {value}")
        
        print("\nLocal Execution:")
        if self.results['local'].get('status') == 'success':
            print(f"  ✓ Status: Success")
            print(f"  Time: {self.results['local']['training_time_minutes']:.2f} min")
            print(f"  Memory: {self.results['local']['memory_used_mb']:.2f} MB")
        else:
            print(f"  ✗ Status: {self.results['local'].get('status', 'unknown')}")
        
        print("\nDocker Execution:")
        if self.results['docker'].get('status') == 'success':
            print(f"  ✓ Status: Success")
            print(f"  Build: {self.results['docker']['build_time_seconds']:.2f} sec")
            print(f"  Training: {self.results['docker']['training_time_minutes']:.2f} min")
            print(f"  Total: {self.results['docker']['total_time_minutes']:.2f} min")
            print(f"  Image: {self.results['docker']['image_size']}")
        else:
            print(f"  ✗ Status: {self.results['docker'].get('status', 'unknown')}")
        
        if 'docker_overhead_percent' in self.results.get('comparison', {}):
            print(f"\nOverhead: {self.results['comparison']['docker_overhead_percent']:+.2f}%")
        
        print("\n" + "="*70)


def main():
    """Main benchmark execution"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   TruthGuard-AI Performance Benchmark                ║
║              Comparing Local vs Docker Training Execution            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks with 1 epoch for faster comparison
    # You can change this to 3 for full comparison
    BENCHMARK_EPOCHS = 1
    
    print(f"Note: Using {BENCHMARK_EPOCHS} epoch(s) for faster benchmarking")
    print("     (Change BENCHMARK_EPOCHS to 3 for full training comparison)")
    
    # Benchmark local execution
    benchmark.measure_local_training(num_epochs=BENCHMARK_EPOCHS)
    
    # Benchmark Docker execution
    benchmark.measure_docker_training(num_epochs=BENCHMARK_EPOCHS)
    
    # Compare results
    benchmark.compare_results()
    
    # Save and print summary
    benchmark.save_results()
    benchmark.print_summary()
    
    print("\n✓ Benchmark completed!")
    print("  Results saved to: logs/benchmark_results.json")


if __name__ == "__main__":
    main()
