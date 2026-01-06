#!/usr/bin/env python3
"""
Symphony Launcher for Llama3-8B with 8 Node Managers

This script orchestrates the entire Symphony system:
- 8 node schedulers (each on a different port)
- 1 global scheduler (load-based scheduling)
- 1 workload generator

Usage:
    python symphony-llama3-8b.py --num-clients 10
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from typing import List

# Configuration for 8 node managers
NODE_MANAGERS = [
    {"id": 0, "port": 9000, "cuda_device": 0},
    {"id": 1, "port": 9001, "cuda_device": 1},
    {"id": 2, "port": 9002, "cuda_device": 2},
    {"id": 3, "port": 9003, "cuda_device": 3},
    {"id": 4, "port": 9004, "cuda_device": 4},
    {"id": 5, "port": 9005, "cuda_device": 5},
    {"id": 6, "port": 9006, "cuda_device": 6},
    {"id": 7, "port": 9007, "cuda_device": 7},
]

# Global scheduler configuration
GLOBAL_SCHEDULER_PORT = 50051
GLOBAL_SCHEDULER_IP = "localhost"

# Workload generator configuration
WORKLOAD_GEN_PORT = 50050
WORKLOAD_GEN_IP = "localhost"


class SymphonyOrchestrator:
    """Orchestrates launching and managing all Symphony components"""
    
    def __init__(self, args):
        self.args = args
        self.processes: List[subprocess.Popen] = []
        self.node_manager_processes: List[subprocess.Popen] = []
        self.scheduler_process = None
        self.workload_gen_process = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle CTRL+C gracefully"""
        print("\n[ORCHESTRATOR] Received interrupt signal, shutting down all components...")
        self.shutdown_all()
        sys.exit(0)
    
    def launch_node_managers(self):
        """Launch all 8 node managers, each on its own GPU"""
        print(f"\n[ORCHESTRATOR] Launching {len(NODE_MANAGERS)} node managers...")
        
        for node in NODE_MANAGERS:
            node_id = node["id"]
            port = node["port"]
            cuda_device = node["cuda_device"]
            
            # Build command for node scheduler
            cmd = [
                "python",
                "node_scheduler.py",
                f"--node-scheduler-port={port}",
                f"--global-scheduler-ip={GLOBAL_SCHEDULER_IP}",
                f"--global-scheduler-port={GLOBAL_SCHEDULER_PORT}",
                f"--model={self.args.model}",
                f"--tensor-parallel-size={self.args.tensor_parallel_size}",
                f"--max-num-seqs={self.args.max_num_seqs}",
                f"--gpu-memory-utilization={self.args.gpu_memory_utilization}",
                f"--memory-threshold={self.args.memory_threshold}",
                f"--cache-eviction-layers={self.args.cache_eviction_layers}",
            ]
            
            # Set environment variables for this process
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
            
            print(f"  [Node {node_id}] Port: {port}, GPU: {cuda_device}")
            print(f"  [Node {node_id}] Command: {' '.join(cmd)}")
            
            # Launch the node scheduler
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.node_manager_processes.append(process)
            self.processes.append(process)
            
            # Give it a moment to start up
            time.sleep(2)
        
        print(f"[ORCHESTRATOR] All {len(NODE_MANAGERS)} node managers launched")
    
    def launch_global_scheduler(self):
        """Launch the global scheduler (load-based scheduling)"""
        print("\n[ORCHESTRATOR] Launching global scheduler...")
        
        cmd = [
            "python",
            "scheduler_based_on_load.py",
            f"--scheduler-port={GLOBAL_SCHEDULER_PORT}",
            f"--workload-gen-ip={WORKLOAD_GEN_IP}",
            f"--workload-gen-port={WORKLOAD_GEN_PORT}",
        ]
        
        print(f"  [Scheduler] Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        self.scheduler_process = process
        self.processes.append(process)
        
        # Give scheduler time to initialize
        time.sleep(3)
        
        print("[ORCHESTRATOR] Global scheduler launched")
    
    def launch_workload_generator(self):
        """Launch the workload generator"""
        print("\n[ORCHESTRATOR] Launching workload generator...")
        
        cmd = [
            "python",
            "workloadgen.py",
            f"--reciever-port={WORKLOAD_GEN_PORT}",
            f"--scheduler-ip={GLOBAL_SCHEDULER_IP}",
            f"--scheduler-port={GLOBAL_SCHEDULER_PORT}",
            f"--num-clients={self.args.num_clients}",
        ]
        
        print(f"  [WorkloadGen] Clients: {self.args.num_clients}")
        print(f"  [WorkloadGen] Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        self.workload_gen_process = process
        self.processes.append(process)
        
        print("[ORCHESTRATOR] Workload generator launched")
    
    def monitor_processes(self):
        """Monitor all processes and print their output"""
        print("\n[ORCHESTRATOR] All components running. Monitoring output...")
        print("[ORCHESTRATOR] Press CTRL+C to stop all processes\n")
        print("="*80)
        
        try:
            while True:
                # Check if any process has terminated unexpectedly
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        # Process has terminated
                        returncode = process.returncode
                        stdout, stderr = process.communicate()
                        
                        print(f"\n[ORCHESTRATOR] Process {i} terminated with code {returncode}")
                        if stdout:
                            print(f"[STDOUT] {stdout}")
                        if stderr:
                            print(f"[STDERR] {stderr}")
                        
                        # If any critical process dies, shut down everything
                        print("[ORCHESTRATOR] Critical process died, shutting down all components")
                        self.shutdown_all()
                        return
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n[ORCHESTRATOR] Keyboard interrupt received")
            self.shutdown_all()
    
    def shutdown_all(self):
        """Shutdown all launched processes"""
        print("\n[ORCHESTRATOR] Shutting down all components...")
        
        # Shutdown in reverse order: workload gen -> scheduler -> node managers
        if self.workload_gen_process:
            print("  [Shutdown] Stopping workload generator...")
            self.workload_gen_process.terminate()
            try:
                self.workload_gen_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.workload_gen_process.kill()
        
        if self.scheduler_process:
            print("  [Shutdown] Stopping global scheduler...")
            self.scheduler_process.terminate()
            try:
                self.scheduler_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.scheduler_process.kill()
        
        for i, process in enumerate(self.node_manager_processes):
            print(f"  [Shutdown] Stopping node manager {i}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("[ORCHESTRATOR] All components shut down")
    
    def run(self):
        """Main orchestration flow"""
        print("="*80)
        print("Symphony Orchestrator - Llama3-8B on 8 GPUs")
        print("="*80)
        print(f"Model: {self.args.model}")
        print(f"Number of clients: {self.args.num_clients}")
        print(f"Tensor parallel size: {self.args.tensor_parallel_size}")
        print(f"Max sequences per node: {self.args.max_num_seqs}")
        print(f"GPU memory utilization: {self.args.gpu_memory_utilization}")
        print(f"Memory threshold: {self.args.memory_threshold}")
        print(f"Cache eviction layers: {self.args.cache_eviction_layers}")
        print("="*80)
        
        try:
            # Launch components in order
            self.launch_node_managers()
            self.launch_global_scheduler()
            self.launch_workload_generator()
            
            # Monitor all processes
            self.monitor_processes()
        
        except Exception as e:
            print(f"\n[ORCHESTRATOR] Error: {e}")
            self.shutdown_all()
            raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Symphony Orchestrator for Llama3-8B with 8 Node Managers"
    )
    
    # Workload generation parameters
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of concurrent client sessions to simulate"
    )
    
    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name"
    )
    
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for each node"
    )
    
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per node"
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)"
    )
    
    # Memory management parameters
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.8,
        help="Memory threshold for cooperative cache eviction (0.0-1.0)"
    )
    
    parser.add_argument(
        "--cache-eviction-layers",
        type=int,
        default=2,
        help="Number of early layers to evict from K,V cache during memory pressure"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    orchestrator = SymphonyOrchestrator(args)
    orchestrator.run()
