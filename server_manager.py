#!/usr/bin/env python3
"""
Server manager for TBot - easily start and manage Conductor and Diffusiond servers
"""

import asyncio
import argparse
import subprocess
import signal
import sys
import time
import os
from pathlib import Path
from typing import List, Optional


class ServerManager:
    """Manages Conductor and Diffusiond servers"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down servers...")
        self.stop_all_servers()
        sys.exit(0)
    
    def start_conductor(self, 
                       config_path: Optional[str] = None,
                       port: int = 8001,
                       log_level: str = "INFO") -> subprocess.Popen:
        """Start the Conductor server"""
        cmd = ["poetry", "run", "python", "-m", "conductor.conductor"]
        
        if config_path:
            cmd.extend(["--config", config_path])
        
        cmd.extend(["--log-level", log_level])
        
        # Set environment variables
        env = os.environ.copy()
        env["CONDUCTOR_PORT"] = str(port)
        
        print(f"🚀 Starting Conductor on port {port}...")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(process)
        return process
    
    def start_diffusiond(self, 
                        model: Optional[str] = None,
                        port: int = 8000,
                        device: str = "auto") -> subprocess.Popen:
        """Start the Diffusiond server"""
        cmd = ["poetry", "run", "python", "-m", "diffusiond.main"]
        
        if model:
            cmd.extend(["--model", model])
        
        cmd.extend(["--device", device])
        
        # Set environment variables
        env = os.environ.copy()
        env["DIFFUSIOND_PORT"] = str(port)
        
        print(f"🎨 Starting Diffusiond on port {port}...")
        if model:
            print(f"   Using model: {model}")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(process)
        return process
    
    def stop_all_servers(self):
        """Stop all running servers"""
        print("🛑 Stopping all servers...")
        for process in self.processes:
            if process.poll() is None:  # Still running
                print(f"   Terminating process {process.pid}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"   Force killing process {process.pid}")
                    process.kill()
        
        self.processes.clear()
        print("✅ All servers stopped")
    
    def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        while self.running:
            for i, process in enumerate(self.processes[:]):
                if process.poll() is not None:  # Process has ended
                    print(f"⚠️  Process {process.pid} has ended with code {process.returncode}")
                    self.processes.remove(process)
            
            time.sleep(1)
    
    def run_both_servers(self, 
                        conductor_config: Optional[str] = None,
                        conductor_port: int = 8001,
                        diffusiond_model: Optional[str] = None,
                        diffusiond_port: int = 8000,
                        device: str = "auto",
                        log_level: str = "INFO"):
        """Run both servers and monitor them"""
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start Conductor
            conductor_proc = self.start_conductor(
                config_path=conductor_config,
                port=conductor_port,
                log_level=log_level
            )
            
            # Start Diffusiond
            diffusiond_proc = self.start_diffusiond(
                model=diffusiond_model,
                port=diffusiond_port,
                device=device
            )
            
            print("\n🎉 Both servers started!")
            print(f"   Conductor: http://localhost:{conductor_port}")
            print(f"   Diffusiond: http://localhost:{diffusiond_port}")
            print("\nPress Ctrl+C to stop all servers")
            print("-" * 50)
            
            # Monitor and output logs
            while self.running and self.processes:
                # Check if any process has output
                for process in self.processes[:]:
                    if process.poll() is not None:
                        print(f"🔄 Process {process.pid} ended, removing from monitoring")
                        self.processes.remove(process)
                        continue
                    
                    # Read any available output
                    if process.stdout and process.stdout.readable():
                        try:
                            line = process.stdout.readline()
                            if line:
                                server_name = "Conductor" if process == conductor_proc else "Diffusiond"
                                print(f"[{server_name}] {line.rstrip()}")
                        except:
                            pass
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all_servers()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TBot Server Manager")
    
    # Server selection
    parser.add_argument("--conductor-only", action="store_true",
                       help="Start only Conductor server")
    parser.add_argument("--diffusiond-only", action="store_true",
                       help="Start only Diffusiond server")
    
    # Conductor options
    parser.add_argument("--conductor-config", 
                       help="Path to Conductor config file")
    parser.add_argument("--conductor-port", type=int, default=8001,
                       help="Conductor server port (default: 8001)")
    
    # Diffusiond options
    parser.add_argument("--diffusiond-model",
                       help="Diffusiond model to use")
    parser.add_argument("--diffusiond-port", type=int, default=8000,
                       help="Diffusiond server port (default: 8000)")
    
    # Common options
    parser.add_argument("--device", default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use (default: auto)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level (default: INFO)")
    
    args = parser.parse_args()
    
    manager = ServerManager()
    
    if args.conductor_only:
        print("🚀 Starting Conductor only...")
        try:
            process = manager.start_conductor(
                config_path=args.conductor_config,
                port=args.conductor_port,
                log_level=args.log_level
            )
            print(f"   Conductor: http://localhost:{args.conductor_port}")
            print("Press Ctrl+C to stop")
            process.wait()
        except KeyboardInterrupt:
            manager.stop_all_servers()
    
    elif args.diffusiond_only:
        print("🎨 Starting Diffusiond only...")
        try:
            process = manager.start_diffusiond(
                model=args.diffusiond_model,
                port=args.diffusiond_port,
                device=args.device
            )
            print(f"   Diffusiond: http://localhost:{args.diffusiond_port}")
            print("Press Ctrl+C to stop")
            process.wait()
        except KeyboardInterrupt:
            manager.stop_all_servers()
    
    else:
        # Start both servers
        manager.run_both_servers(
            conductor_config=args.conductor_config,
            conductor_port=args.conductor_port,
            diffusiond_model=args.diffusiond_model,
            diffusiond_port=args.diffusiond_port,
            device=args.device,
            log_level=args.log_level
        )


if __name__ == "__main__":
    main()
