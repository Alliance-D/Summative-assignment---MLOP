"""
Automated Load Testing with Docker Containers
Tests API performance with 1, 2, and 4 Docker containers
"""

import subprocess
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os

# Configuration
CONTAINER_CONFIGS = [
    {"containers": 1, "compose_file": "docker-compose-1container.yml"},
    {"containers": 2, "compose_file": "docker-compose-2containers.yml"},
    {"containers": 4, "compose_file": "docker-compose.yml"},  # Main file with 4 containers
]

TEST_DURATION = "2m"
USERS = 50
SPAWN_RATE = 5
RESULTS_DIR = Path("docker_load_test_results")

RESULTS_DIR.mkdir(exist_ok=True)


class DockerLoadTester:
    """Manages Docker-based load testing"""
    
    def __init__(self):
        self.results = []
    
    def build_image(self):
        """Build Docker image once"""
        print("\n" + "="*70)
        print("🐳 Building Docker Image")
        print("="*70)
        
        cmd = ["docker", "build", "-t", "plant-disease-api", "."]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker image built successfully")
            return True
        else:
            print("❌ Docker build failed:")
            print(result.stderr)
            return False
    
    def start_containers(self, compose_file: str, num_containers: int):
        """Start Docker containers"""
        print(f"\n{'='*70}")
        print(f"🚀 Starting {num_containers} Container(s)")
        print(f"{'='*70}")
        
        # Stop any existing containers
        self.stop_containers(compose_file)
        time.sleep(5)
        
        cmd = ["docker-compose", "-f", compose_file, "up", "-d"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Failed to start containers:")
            print(result.stderr)
            return False
        
        print(f"⏳ Waiting for containers to be ready...")
        time.sleep(30)  # Wait for containers to fully start
        
        # Verify containers are running
        if self.verify_containers_running(num_containers):
            print("✅ All containers are running")
            return True
        else:
            print("❌ Some containers failed to start")
            return False
    
    def verify_containers_running(self, expected_count: int):
        """Verify all containers are healthy"""
        cmd = ["docker", "ps", "--filter", "name=plant-disease-api", "--format", "{{.Names}}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        running_containers = [line for line in result.stdout.strip().split('\n') if line]
        return len(running_containers) >= expected_count
    
    def stop_containers(self, compose_file: str):
        """Stop and remove containers"""
        print(f"\n🛑 Stopping containers...")
        
        cmd = ["docker-compose", "-f", compose_file, "down"]
        subprocess.run(cmd, capture_output=True, text=True)
        
        print("✅ Containers stopped")
        time.sleep(5)
    
    def run_locust_test(self, num_containers: int):
        """Run Locust load test"""
        print(f"\n{'='*70}")
        print(f"🧪 Running Load Test - {num_containers} Container(s)")
        print(f"{'='*70}")
        print(f"Users: {USERS}")
        print(f"Spawn Rate: {SPAWN_RATE}/s")
        print(f"Duration: {TEST_DURATION}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_report = RESULTS_DIR / f"report_{num_containers}containers_{timestamp}.html"
        csv_report = RESULTS_DIR / f"stats_{num_containers}containers_{timestamp}.csv"
        
        cmd = [
            "locust",
            "-f", "locustfile.py",
            "--host", "http://localhost:8000",
            "--users", str(USERS),
            "--spawn-rate", str(SPAWN_RATE),
            "--run-time", TEST_DURATION,
            "--headless",
            "--html", str(html_report),
            "--csv", str(csv_report.with_suffix(''))
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse results
            stats = self._parse_locust_output(result.stdout)
            stats['containers'] = num_containers
            stats['timestamp'] = timestamp
            self.results.append(stats)
            
            print(f"\n✅ Test completed for {num_containers} container(s)")
            print(f"📊 Average Response Time: {stats['avg_response_time']:.2f}ms")
            print(f"📊 Requests/sec: {stats['requests_per_sec']:.2f}")
            
            return stats
            
        except subprocess.TimeoutExpired:
            print(f"⚠️ Test timed out for {num_containers} containers")
            return None
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            return None
    
    def _parse_locust_output(self, output: str) -> dict:
        """Parse Locust output"""
        stats = {
            'avg_response_time': 0,
            'median_response_time': 0,
            'p95_response_time': 0,
            'p99_response_time': 0,
            'requests_per_sec': 0,
            'failure_rate': 0,
            'total_requests': 0
        }
        
        lines = output.split('\n')
        for line in lines:
            # Look for aggregated stats line
            if 'Aggregated' in line or '/predict' in line:
                parts = line.split()
                try:
                    # Extract metrics based on position
                    for i, part in enumerate(parts):
                        if part.replace('.', '').isdigit():
                            if 'avg' not in stats or stats['avg_response_time'] == 0:
                                stats['avg_response_time'] = float(part)
                            elif 'median' not in stats or stats['median_response_time'] == 0:
                                stats['median_response_time'] = float(part)
                except (IndexError, ValueError):
                    pass
            
            # Extract RPS
            if 'RPS' in line or 'requests/s' in line.lower():
                parts = line.split()
                for part in parts:
                    if part.replace('.', '').isdigit():
                        stats['requests_per_sec'] = float(part)
                        break
        
        return stats
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*70)
        print("🐳 DOCKER LOAD TESTING SUITE")
        print("="*70)
        print(f"\nTesting: {[c['containers'] for c in CONTAINER_CONFIGS]} containers")
        print(f"Estimated time: {len(CONTAINER_CONFIGS) * 4} minutes")
        
        # Build image first
        if not self.build_image():
            print("❌ Failed to build Docker image. Exiting.")
            return
        
        # Run tests for each configuration
        for config in CONTAINER_CONFIGS:
            num_containers = config['containers']
            compose_file = config['compose_file']
            
            try:
                # Start containers
                if not self.start_containers(compose_file, num_containers):
                    print(f"⚠️ Skipping test for {num_containers} containers")
                    continue
                
                # Wait a bit more for stability
                time.sleep(10)
                
                # Run load test
                self.run_locust_test(num_containers)
                
                # Stop containers
                self.stop_containers(compose_file)
                
                # Wait between tests
                time.sleep(10)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Tests interrupted by user")
                self.stop_containers(compose_file)
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                self.stop_containers(compose_file)
                continue
        
        # Generate report
        if self.results:
            self.generate_report()
        else:
            print("\n⚠️ No results to generate report")
    
    def generate_report(self):
        """Generate comparison report"""
        print("\n" + "="*70)
        print("📊 GENERATING REPORTS")
        print("="*70)
        
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_file = RESULTS_DIR / f"docker_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✅ CSV saved: {csv_file}")
        
        # Generate charts
        self._create_charts(df)
        
        # Print summary
        self._print_summary(df)
    
    def _create_charts(self, df: pd.DataFrame):
        """Create comparison charts"""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Chart 1: Response Time
        ax1 = axes[0, 0]
        containers = df['containers']
        ax1.plot(containers, df['avg_response_time'], marker='o', linewidth=2, markersize=10, color='blue')
        ax1.set_xlabel('Number of Docker Containers', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Response Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Response Time vs Docker Containers', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(containers)
        
        # Chart 2: Throughput
        ax2 = axes[0, 1]
        ax2.bar(containers, df['requests_per_sec'], color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Docker Containers', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Requests Per Second', fontsize=12, fontweight='bold')
        ax2.set_title('Throughput vs Docker Containers', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(containers)
        
        # Chart 3: Speedup
        ax3 = axes[1, 0]
        baseline_rps = df['requests_per_sec'].iloc[0]
        speedup = df['requests_per_sec'] / baseline_rps
        ax3.plot(containers, speedup, marker='s', linewidth=2, markersize=10, color='red')
        ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        # Add ideal linear scaling line
        ideal_speedup = containers / containers.iloc[0]
        ax3.plot(containers, ideal_speedup, linestyle='--', color='orange', label='Ideal Linear Scaling', linewidth=2)
        
        ax3.set_xlabel('Number of Docker Containers', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
        ax3.set_title('Performance Speedup vs Docker Containers', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(containers)
        
        # Chart 4: Latency Reduction
        ax4 = axes[1, 1]
        baseline_latency = df['avg_response_time'].iloc[0]
        latency_reduction = ((baseline_latency - df['avg_response_time']) / baseline_latency) * 100
        colors = ['red' if x < 0 else 'green' for x in latency_reduction]
        ax4.bar(containers, latency_reduction, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Number of Docker Containers', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Latency Reduction (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Latency Improvement vs Docker Containers', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticks(containers)
        
        plt.tight_layout()
        chart_file = RESULTS_DIR / f"docker_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"✅ Charts saved: {chart_file}")
        
        plt.show()
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary"""
        print("\n" + "="*70)
        print("📈 DOCKER SCALING PERFORMANCE SUMMARY")
        print("="*70)
        
        print(f"\n{'Containers':<12} {'Avg RT (ms)':<15} {'RPS':<15} {'Speedup':<10}")
        print("-" * 70)
        
        baseline_rps = df['requests_per_sec'].iloc[0]
        
        for _, row in df.iterrows():
            speedup = row['requests_per_sec'] / baseline_rps
            print(f"{row['containers']:<12} {row['avg_response_time']:<15.2f} "
                  f"{row['requests_per_sec']:<15.2f} {speedup:<10.2f}x")
        
        print("\n" + "="*70)
        print("🎯 KEY FINDINGS:")
        print("="*70)
        
        # Best performance
        best_throughput = df.loc[df['requests_per_sec'].idxmax()]
        best_latency = df.loc[df['avg_response_time'].idxmin()]
        
        print(f"\n✅ Highest Throughput: {best_throughput['containers']} containers "
              f"({best_throughput['requests_per_sec']:.2f} RPS)")
        print(f"✅ Lowest Latency: {best_latency['containers']} containers "
              f"({best_latency['avg_response_time']:.2f} ms)")
        
        # Scaling efficiency
        final_speedup = df['requests_per_sec'].iloc[-1] / baseline_rps
        ideal_speedup = df['containers'].iloc[-1] / df['containers'].iloc[0]
        efficiency = (final_speedup / ideal_speedup) * 100
        
        print(f"\n📊 Scaling Efficiency: {efficiency:.1f}%")
        print(f"   Achieved: {final_speedup:.2f}x speedup")
        print(f"   Ideal: {ideal_speedup:.0f}x speedup")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🐳 DOCKER-BASED LOAD TESTING")
    print("="*70)
    print("\nThis will test your API with:")
    print("  • 1 Docker container")
    print("  • 2 Docker containers + Nginx load balancer")
    print("  • 4 Docker containers + Nginx load balancer")
    print(f"\nEstimated duration: {len(CONTAINER_CONFIGS) * 4} minutes")
    
    print("\n⚠️  Prerequisites:")
    print("  • Docker installed and running")
    print("  • docker-compose installed")
    print("  • Port 8000 available")
    print("  • Trained model in models/ directory")
    
    input("\nPress ENTER to start (or CTRL+C to cancel)...")
    
    tester = DockerLoadTester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted")
    finally:
        print("\n✅ Testing complete!")