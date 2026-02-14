"""
Deployment Service - Handles deploying generated apps to EC2 instance
"""
import os
import subprocess
import tempfile
import tarfile
import io
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple


class DeploymentService:
    def __init__(self, target_ec2_ip: str, ssh_key_path: str, remote_user: str = "ubuntu"):
        """
        Initialize deployment service.
        
        Args:
            target_ec2_ip: IP address of EC2 instance where apps will be deployed
            ssh_key_path: Path to SSH private key for EC2 access
            remote_user: SSH user (default: ubuntu)
        """
        self.target_ec2_ip = target_ec2_ip
        self.ssh_key_path = ssh_key_path
        self.remote_user = remote_user
        self.remote_apps_dir = "/var/www/deployed-apps"
        self.base_frontend_port = 3000
        self.base_backend_port = 8000
        
        # Validate SSH key exists and has correct permissions
        self._validate_ssh_key()
    
    def _validate_ssh_key(self):
        """Validate SSH key file exists and has correct permissions."""
        import stat
        
        if not os.path.exists(self.ssh_key_path):
            raise Exception(f"SSH key not found at: {self.ssh_key_path}")
        
        # Check file permissions (should be 400 or 600)
        file_stat = os.stat(self.ssh_key_path)
        file_mode = stat.S_IMODE(file_stat.st_mode)
        
        # If permissions are too open, try to fix them
        if file_mode not in (0o400, 0o600):
            try:
                os.chmod(self.ssh_key_path, 0o600)
                print(f"Fixed SSH key permissions: {self.ssh_key_path}")
            except Exception as e:
                print(f"Warning: Could not fix SSH key permissions: {e}")

    def find_available_ports(self) -> Tuple[int, int]:
        """
        Find available ports on the remote EC2 instance.
        Returns: (frontend_port, backend_port)
        """
        try:
            # Check which ports are in use
            cmd = [
                "ssh", "-i", self.ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                f"{self.remote_user}@{self.target_ec2_ip}",
                "netstat -tuln | grep LISTEN || true"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            used_ports = set()
            
            for line in result.stdout.split('\n'):
                if ':' in line:
                    parts = line.split()
                    for part in parts:
                        if ':' in part:
                            try:
                                port = int(part.split(':')[-1])
                                used_ports.add(port)
                            except ValueError:
                                continue
            
            # Find available ports
            frontend_port = self.base_frontend_port
            while frontend_port in used_ports:
                frontend_port += 1
            
            backend_port = self.base_backend_port
            while backend_port in used_ports:
                backend_port += 1
            
            return frontend_port, backend_port
            
        except Exception as e:
            print(f"Error finding ports: {e}, using defaults")
            return self.base_frontend_port, self.base_backend_port

    def deploy_app(self, project_name: str, project_path: str) -> Dict[str, any]:
        """
        Deploy an app to EC2 instance.
        
        Args:
            project_name: Name of the project
            project_path: Local path to the project files
            
        Returns:
            Dict with deployment info: {
                'frontend_url': str,
                'backend_url': str,
                'frontend_port': int,
                'backend_port': int,
                'status': 'success' | 'error',
                'message': str
            }
        """
        try:
            # Find available ports
            frontend_port, backend_port = self.find_available_ports()
            
            # Create tar archive of the project
            tar_path = self._create_project_archive(project_path, project_name)
            
            # Setup remote directory
            self._setup_remote_directory(project_name)
            
            # Upload project files
            self._upload_project(tar_path, project_name)
            
            # Extract on remote
            self._extract_remote_archive(project_name)
            
            # Stop any existing deployment for this project
            self._stop_existing_deployment(project_name)
            
            # Install dependencies and start the app
            self._start_app(project_name, frontend_port, backend_port)
            
            # Clean up local tar file
            os.remove(tar_path)
            
            return {
                'frontend_url': f"http://{self.target_ec2_ip}:{frontend_port}",
                'backend_url': f"http://{self.target_ec2_ip}:{backend_port}",
                'frontend_port': frontend_port,
                'backend_port': backend_port,
                'status': 'success',
                'message': 'Deployment successful'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def _create_project_archive(self, project_path: str, project_name: str) -> str:
        """Create a tar.gz archive of the project."""
        tar_path = os.path.join(tempfile.gettempdir(), f"{project_name}.tar.gz")
        
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(project_path, arcname=project_name)
        
        return tar_path

    def _setup_remote_directory(self, project_name: str):
        """Create remote directory structure."""
        # First check SSH connectivity
        try:
            test_cmd = [
                "ssh", "-i", self.ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                f"{self.remote_user}@{self.target_ec2_ip}",
                "echo 'SSH connection successful'"
            ]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                raise Exception(f"SSH connection failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise Exception("SSH connection timeout - check EC2 instance and security groups")
        except Exception as e:
            raise Exception(f"SSH connection error: {str(e)}")
        
        # Create directory with proper permissions
        cmd = [
            "ssh", "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            f"{self.remote_user}@{self.target_ec2_ip}",
            f"sudo mkdir -p {self.remote_apps_dir} && sudo chown -R {self.remote_user}:{self.remote_user} {self.remote_apps_dir}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise Exception(f"Failed to setup remote directory: {result.stderr}")

    def _upload_project(self, tar_path: str, project_name: str):
        """Upload project archive to EC2."""
        cmd = [
            "scp", "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            tar_path,
            f"{self.remote_user}@{self.target_ec2_ip}:{self.remote_apps_dir}/{project_name}.tar.gz"
        ]
        subprocess.run(cmd, check=True, timeout=120)

    def _extract_remote_archive(self, project_name: str):
        """Extract the uploaded archive on EC2."""
        cmd = [
            "ssh", "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            f"{self.remote_user}@{self.target_ec2_ip}",
            f"cd {self.remote_apps_dir} && rm -rf {project_name} && tar -xzf {project_name}.tar.gz && rm {project_name}.tar.gz"
        ]
        subprocess.run(cmd, check=True, timeout=60)

    def _stop_existing_deployment(self, project_name: str):
        """Stop any existing processes for this project."""
        # Kill processes by project directory name
        cmd = [
            "ssh", "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            f"{self.remote_user}@{self.target_ec2_ip}",
            f"pkill -f '{self.remote_apps_dir}/{project_name}' || true"
        ]
        subprocess.run(cmd, timeout=30)

    def _start_app(self, project_name: str, frontend_port: int, backend_port: int):
        """Install dependencies and start the app."""
        project_dir = f"{self.remote_apps_dir}/{project_name}"
        
        # Check if project has frontend and/or backend
        check_cmd = [
            "ssh", "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            f"{self.remote_user}@{self.target_ec2_ip}",
            f"ls -d {project_dir}/{{frontend,backend}} 2>/dev/null || true"
        ]
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
        has_frontend = f"{project_dir}/frontend" in result.stdout
        has_backend = f"{project_dir}/backend" in result.stdout
        
        # Start backend if exists
        if has_backend:
            backend_cmd = f"""
cd {project_dir}/backend && \
python3 -m venv venv 2>/dev/null || true && \
(source venv/bin/activate 2>/dev/null || true) && \
(pip3 install -r requirements.txt 2>/dev/null || true) && \
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port {backend_port} > /tmp/{project_name}_backend.log 2>&1 &
"""
            cmd = [
                "ssh", "-i", self.ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                f"{self.remote_user}@{self.target_ec2_ip}",
                backend_cmd
            ]
            subprocess.run(cmd, timeout=180)
        
        # Start frontend if exists
        if has_frontend:
            frontend_cmd = f"""
cd {project_dir}/frontend && \
(npm install 2>/dev/null || true) && \
nohup npm run dev -- --host 0.0.0.0 --port {frontend_port} > /tmp/{project_name}_frontend.log 2>&1 &
"""
            cmd = [
                "ssh", "-i", self.ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                f"{self.remote_user}@{self.target_ec2_ip}",
                frontend_cmd
            ]
            subprocess.run(cmd, timeout=300)

    def check_deployment_health(self, frontend_port: int, backend_port: int) -> Dict[str, bool]:
        """Check if deployed services are running."""
        import requests
        
        result = {
            'frontend': False,
            'backend': False
        }
        
        try:
            # Check frontend
            response = requests.get(f"http://{self.target_ec2_ip}:{frontend_port}", timeout=5)
            result['frontend'] = response.status_code < 500
        except Exception:
            pass
        
        try:
            # Check backend (try /docs endpoint for FastAPI)
            response = requests.get(f"http://{self.target_ec2_ip}:{backend_port}/docs", timeout=5)
            result['backend'] = response.status_code < 500
        except Exception:
            pass
        
        return result
