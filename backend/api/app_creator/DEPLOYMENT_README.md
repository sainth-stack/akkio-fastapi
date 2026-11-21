# Deployment & GitHub Integration Setup

This guide covers setting up EC2 deployment and GitHub integration for the Akkio App Builder.

## Table of Contents

1. [EC2 Deployment Setup](#ec2-deployment-setup)
2. [GitHub Integration Setup](#github-integration-setup)
3. [Troubleshooting](#troubleshooting)

---

## EC2 Deployment Setup

### Prerequisites

1. An AWS EC2 instance with:
   - Ubuntu/Amazon Linux OS
   - Python 3.8+ installed
   - Node.js 16+ and npm installed
   - Port 3000-3100 and 8000-8100 open in security groups
   - SSH access configured

2. SSH key pair for EC2 access

### Environment Variables

Add these to your `.env` file:

```bash
# EC2 Deployment Configuration
DEPLOY_EC2_IP=54.151.251.149           # Your EC2 instance IP
DEPLOY_SSH_KEY_PATH=/path/to/key.pem   # Path to your SSH private key
DEPLOY_SSH_USER=ubuntu                  # SSH user (ubuntu, ec2-user, or root)
```

### SSH Key Setup

1. Ensure your SSH key has correct permissions:
   ```bash
   chmod 600 /path/to/your-key.pem
   ```

2. Test SSH connection:
   ```bash
   ssh -i /path/to/your-key.pem ubuntu@YOUR_EC2_IP
   ```

### Common SSH Issues

#### Exit Status 255 Error

This error typically means SSH connection failed. Check:

1. **Security Groups**: Ensure port 22 is open for your IP
   - Go to EC2 Console → Security Groups
   - Add inbound rule: SSH (22) from your IP

2. **SSH Key Permissions**: Key file must be 600
   ```bash
   ls -l /path/to/key.pem
   # Should show: -rw------- (600)
   ```

3. **Correct User**: Different AMIs use different users:
   - Ubuntu AMI: `ubuntu`
   - Amazon Linux: `ec2-user`
   - Some custom AMIs: `root`

4. **EC2 Instance State**: Ensure instance is running
   ```bash
   aws ec2 describe-instances --instance-ids i-xxxxx
   ```

5. **Network Connectivity**: Try pinging the instance
   ```bash
   ping YOUR_EC2_IP
   ```

### EC2 Instance Preparation

SSH into your EC2 instance and run:

```bash
# Update system
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
# OR
sudo yum update -y                       # Amazon Linux

# Install Python 3 and pip
sudo apt install python3 python3-pip python3-venv -y  # Ubuntu
# OR
sudo yum install python3 python3-pip -y                 # Amazon Linux

# Install Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs  # Ubuntu
# OR
sudo yum install -y nodejs  # Amazon Linux

# Create deployment directory
sudo mkdir -p /var/www/deployed-apps
sudo chown -R ubuntu:ubuntu /var/www/deployed-apps  # Use your SSH user

# Install process manager (optional but recommended)
sudo npm install -g pm2
```

---

## GitHub Integration Setup

### Prerequisites

1. GitHub account
2. GitHub CLI (`gh`) installed on the deployment server (optional)
3. GitHub Personal Access Token (for API access)

### Environment Variables

Add to your `.env` file:

```bash
# GitHub Configuration
GITHUB_TOKEN=ghp_your_token_here  # Optional: for creating repos via API
```

### GitHub CLI Setup (Recommended)

Install GitHub CLI for easier repository management:

```bash
# macOS
brew install gh

# Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate
gh auth login
```

### GitHub Personal Access Token

If not using GitHub CLI:

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic) with these scopes:
   - `repo` (full control of private repositories)
   - `workflow` (if you want to trigger actions)
3. Copy the token and add to `.env`

### Git Configuration

Configure git on the server:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# For HTTPS authentication with token
git config --global credential.helper store
echo "https://USERNAME:TOKEN@github.com" > ~/.git-credentials
```

---

## API Usage

### Deploy to EC2

```bash
POST /api/deployment/deploy
Content-Type: application/json

{
  "app_id": 123,              # Optional
  "project_name": "my-app",
  "user_email": "user@example.com"
}
```

### Check Deployment Status

```bash
GET /api/deployment/status?project_name=my-app&user_email=user@example.com
```

### Push to GitHub

```bash
POST /api/github/push
Content-Type: application/json

{
  "project_name": "my-app",
  "repo_url": "https://github.com/username/repo.git",
  "branch": "main",
  "commit_message": "Deploy from Akkio"
}
```

### Create New Repository and Push

```bash
POST /api/github/push
Content-Type: application/json

{
  "project_name": "my-app",
  "create_repo": true,
  "repo_name": "username/my-app",
  "branch": "main"
}
```

### Get Git Status

```bash
GET /api/github/status?project_name=my-app
```

---

## Troubleshooting

### Deployment Issues

#### SSH Connection Timeout

**Problem**: SSH connection times out

**Solutions**:
1. Check security group rules allow SSH (port 22) from your IP
2. Verify EC2 instance is running
3. Check VPC/subnet configuration
4. Try increasing timeout in code

#### Permission Denied

**Problem**: SSH key is rejected

**Solutions**:
1. Verify key permissions: `chmod 600 key.pem`
2. Ensure you're using the correct key for the instance
3. Check you're using the right user (ubuntu/ec2-user/root)
4. Regenerate key pair if needed

#### Port Already in Use

**Problem**: Deployment fails because port is busy

**Solutions**:
1. Stop existing processes: `pkill -f "project-name"`
2. Change base ports in `deployment_service.py`
3. Use process manager like PM2 to manage apps

### GitHub Issues

#### Authentication Failed

**Problem**: Git push fails with authentication error

**Solutions**:
1. Install and authenticate GitHub CLI: `gh auth login`
2. Use Personal Access Token in git credentials
3. Use SSH keys instead of HTTPS:
   ```bash
   git remote set-url origin git@github.com:username/repo.git
   ```

#### Repository Already Exists

**Problem**: Trying to create a repo that exists

**Solutions**:
1. Use existing repo URL instead of creating new one
2. Delete or rename the existing repository
3. Push to a different repository name

#### Git Not Initialized

**Problem**: Git operations fail because repo isn't initialized

**Solutions**:
1. Call `/api/github/init` endpoint first
2. Or use the push endpoint which auto-initializes

### Network Issues

#### Cannot Reach EC2

**Problem**: Deployment server can't reach EC2

**Solutions**:
1. Check firewall rules on deployment server
2. Verify EC2 public IP hasn't changed
3. Test with: `telnet EC2_IP 22`
4. Check AWS service status

#### Slow Uploads

**Problem**: Project upload takes too long

**Solutions**:
1. Exclude node_modules and large files
2. Use .gitignore to skip unnecessary files
3. Compress project better
4. Consider using S3 for large files

---

## Best Practices

1. **Security**:
   - Never commit SSH keys or tokens to git
   - Use environment variables for sensitive data
   - Rotate credentials regularly
   - Use IAM roles on EC2 instead of keys when possible

2. **Monitoring**:
   - Check deployment logs regularly
   - Use PM2 for process management
   - Set up CloudWatch for EC2 monitoring
   - Monitor port usage

3. **Git Workflow**:
   - Always commit before deploying
   - Use descriptive commit messages
   - Tag releases
   - Keep main branch deployable

4. **Deployment**:
   - Test locally before deploying
   - Keep deployment directory organized
   - Clean up old deployments
   - Document environment-specific configs

---

## Support

For issues:
1. Check logs in `/tmp/{project_name}_frontend.log` and `/tmp/{project_name}_backend.log`
2. Review API error responses
3. Verify environment variables are set correctly
4. Test SSH connection manually

Common log locations:
- FastAPI logs: Check your uvicorn output
- EC2 deployment logs: `/tmp/{project_name}_*.log`
- Git operations: Use `--verbose` flag for debugging
