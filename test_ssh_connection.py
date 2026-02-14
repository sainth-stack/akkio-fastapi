#!/usr/bin/env python3
"""
Test SSH connection from deployment service
Run this on FastAPI EC2 to debug SSH issues
"""
import os
import subprocess
import sys

print("=" * 60)
print("SSH Connection Test")
print("=" * 60)

# Check environment variables
target_ip = os.environ.get("DEPLOY_EC2_IP")
ssh_key = os.environ.get("DEPLOY_SSH_KEY_PATH")
ssh_user = os.environ.get("DEPLOY_SSH_USER")

print(f"\n📋 Environment Variables:")
print(f"   DEPLOY_EC2_IP: {target_ip or '❌ NOT SET'}")
print(f"   DEPLOY_SSH_KEY_PATH: {ssh_key or '❌ NOT SET'}")
print(f"   DEPLOY_SSH_USER: {ssh_user or '❌ NOT SET'}")

if not all([target_ip, ssh_key, ssh_user]):
    print("\n❌ Missing environment variables!")
    print("\nSet them with:")
    print("export DEPLOY_EC2_IP=54.151.251.149")
    print("export DEPLOY_SSH_KEY_PATH=/root/.ssh/deploy-key.pem")
    print("export DEPLOY_SSH_USER=root")
    sys.exit(1)

# Check if key file exists
print(f"\n📁 SSH Key File:")
if os.path.exists(ssh_key):
    print(f"   ✅ File exists: {ssh_key}")
    # Check permissions
    stat = os.stat(ssh_key)
    perms = oct(stat.st_mode)[-3:]
    print(f"   Permissions: {perms}")
    if perms != "600":
        print(f"   ⚠️  Should be 600! Run: chmod 600 {ssh_key}")
else:
    print(f"   ❌ File not found: {ssh_key}")
    sys.exit(1)

# Test SSH connection
print(f"\n🔌 Testing SSH Connection:")
print(f"   ssh -i {ssh_key} {ssh_user}@{target_ip}")

cmd = [
    "ssh", "-i", ssh_key,
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    f"{ssh_user}@{target_ip}",
    "echo 'SSH connection successful!'"
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    
    if result.returncode == 0:
        print(f"\n✅ SSH Connection Successful!")
        print(f"   Output: {result.stdout.strip()}")
    else:
        print(f"\n❌ SSH Connection Failed!")
        print(f"   Exit Code: {result.returncode}")
        print(f"   Error: {result.stderr}")
        
        # Common issues
        print(f"\n💡 Troubleshooting:")
        if result.returncode == 255:
            print("   - Exit code 255 usually means:")
            print("     1. Wrong IP address")
            print("     2. SSH key doesn't match")
            print("     3. Network/firewall blocking connection")
            print("     4. Target server is down")
        
        print(f"\n🔍 Try manual connection:")
        print(f"   ssh -i {ssh_key} -v {ssh_user}@{target_ip}")
        
except subprocess.TimeoutExpired:
    print(f"\n❌ SSH Connection Timed Out!")
    print(f"   The target server may be unreachable")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
