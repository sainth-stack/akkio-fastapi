"""
GitHub Service - Handles pushing generated apps to GitHub repositories
"""
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional


class GitHubService:
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitHub service.
        
        Args:
            github_token: GitHub personal access token for authentication
        """
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
    
    def initialize_git_repo(self, project_path: str) -> Dict[str, any]:
        """
        Initialize a git repository in the project directory.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Dict with status and message
        """
        try:
            # Check if git is already initialized
            git_dir = os.path.join(project_path, ".git")
            if os.path.exists(git_dir):
                return {
                    'status': 'success',
                    'message': 'Git repository already initialized'
                }
            
            # Initialize git repository
            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Create .gitignore if it doesn't exist
            gitignore_path = os.path.join(project_path, ".gitignore")
            if not os.path.exists(gitignore_path):
                gitignore_content = """# Dependencies
node_modules/
__pycache__/
*.pyc
venv/
env/
.env
.venv

# Build outputs
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*
"""
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content)
            
            return {
                'status': 'success',
                'message': 'Git repository initialized successfully'
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'status': 'error',
                'message': f'Failed to initialize git: {e.stderr}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error initializing git: {str(e)}'
            }
    
    def commit_changes(self, project_path: str, commit_message: str = "Initial commit from Akkio App Builder") -> Dict[str, any]:
        """
        Commit all changes in the project.
        
        Args:
            project_path: Path to the project directory
            commit_message: Commit message
            
        Returns:
            Dict with status and message
        """
        try:
            # Configure git user if not already configured
            try:
                subprocess.run(
                    ["git", "config", "user.email"],
                    cwd=project_path,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError:
                # Set default git config
                subprocess.run(
                    ["git", "config", "user.email", "akkio-builder@example.com"],
                    cwd=project_path,
                    check=True
                )
                subprocess.run(
                    ["git", "config", "user.name", "Akkio App Builder"],
                    cwd=project_path,
                    check=True
                )
            
            # Add all files
            subprocess.run(
                ["git", "add", "-A"],
                cwd=project_path,
                check=True,
                capture_output=True
            )
            
            # Commit changes
            result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=project_path,
                capture_output=True,
                text=True
            )
            
            # Check if there were changes to commit
            if result.returncode != 0:
                if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                    return {
                        'status': 'success',
                        'message': 'No changes to commit'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f'Failed to commit: {result.stderr}'
                    }
            
            return {
                'status': 'success',
                'message': 'Changes committed successfully'
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'status': 'error',
                'message': f'Failed to commit changes: {e.stderr}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error committing changes: {str(e)}'
            }
    
    def create_github_repo(self, repo_name: str, description: str = "", private: bool = True) -> Dict[str, any]:
        """
        Create a new GitHub repository using GitHub CLI or API.
        
        Args:
            repo_name: Name of the repository
            description: Repository description
            private: Whether the repository should be private
            
        Returns:
            Dict with status, message, and repo_url
        """
        try:
            # Try using GitHub CLI first (gh)
            visibility = "private" if private else "public"
            cmd = [
                "gh", "repo", "create", repo_name,
                f"--{visibility}",
                "--description", description or f"Generated by Akkio App Builder"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Extract repo URL from output
                repo_url = result.stdout.strip()
                if not repo_url.startswith("http"):
                    # Parse output to find the URL
                    for line in result.stdout.split('\n'):
                        if 'https://github.com' in line:
                            repo_url = line.strip()
                            break
                    else:
                        # Construct URL manually
                        repo_url = f"https://github.com/{repo_name}"
                
                return {
                    'status': 'success',
                    'message': 'GitHub repository created successfully',
                    'repo_url': repo_url
                }
            else:
                # Check if repo already exists
                if "already exists" in result.stderr.lower():
                    return {
                        'status': 'error',
                        'message': f'Repository {repo_name} already exists on GitHub'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f'Failed to create GitHub repository: {result.stderr}'
                    }
        
        except FileNotFoundError:
            return {
                'status': 'error',
                'message': 'GitHub CLI (gh) not installed. Please install it or provide a repository URL.'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error creating GitHub repository: {str(e)}'
            }
    
    def push_to_github(
        self, 
        project_path: str, 
        repo_url: str,
        branch: str = "main",
        create_repo: bool = False,
        repo_name: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Push project to GitHub repository.
        
        Args:
            project_path: Path to the project directory
            repo_url: GitHub repository URL (or None if creating new)
            branch: Branch name to push to
            create_repo: Whether to create a new repository
            repo_name: Name for new repository (required if create_repo=True)
            
        Returns:
            Dict with status, message, and repo_url
        """
        try:
            # If creating a new repo, do that first
            if create_repo:
                if not repo_name:
                    return {
                        'status': 'error',
                        'message': 'repo_name is required when create_repo=True'
                    }
                
                create_result = self.create_github_repo(repo_name)
                if create_result['status'] != 'success':
                    return create_result
                
                repo_url = create_result['repo_url']
            
            # Initialize git if not already done
            init_result = self.initialize_git_repo(project_path)
            if init_result['status'] != 'success':
                return init_result
            
            # Commit changes
            commit_result = self.commit_changes(project_path)
            if commit_result['status'] != 'success' and 'No changes to commit' not in commit_result['message']:
                return commit_result
            
            # Add remote if not already added
            try:
                subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=project_path,
                    check=True,
                    capture_output=True
                )
                # Remote exists, update it
                subprocess.run(
                    ["git", "remote", "set-url", "origin", repo_url],
                    cwd=project_path,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError:
                # Remote doesn't exist, add it
                subprocess.run(
                    ["git", "remote", "add", "origin", repo_url],
                    cwd=project_path,
                    check=True,
                    capture_output=True
                )
            
            # Rename branch to specified branch if needed
            current_branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=project_path,
                capture_output=True,
                text=True
            )
            current_branch = current_branch_result.stdout.strip()
            
            if current_branch and current_branch != branch:
                subprocess.run(
                    ["git", "branch", "-M", branch],
                    cwd=project_path,
                    check=True,
                    capture_output=True
                )
            
            # Push to GitHub
            push_result = subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if push_result.returncode != 0:
                # Check for common errors
                if "rejected" in push_result.stderr.lower():
                    return {
                        'status': 'error',
                        'message': 'Push rejected. Repository may have changes. Try pulling first or use force push.'
                    }
                elif "authentication" in push_result.stderr.lower() or "permission denied" in push_result.stderr.lower():
                    return {
                        'status': 'error',
                        'message': 'Authentication failed. Please configure GitHub credentials (gh auth login or git credentials).'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f'Failed to push to GitHub: {push_result.stderr}'
                    }
            
            return {
                'status': 'success',
                'message': 'Successfully pushed to GitHub',
                'repo_url': repo_url,
                'branch': branch
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'status': 'error',
                'message': f'Git command failed: {e.stderr if hasattr(e, "stderr") else str(e)}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error pushing to GitHub: {str(e)}'
            }
    
    def get_repo_status(self, project_path: str) -> Dict[str, any]:
        """
        Get the status of the git repository.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Dict with repository information
        """
        try:
            # Check if git is initialized
            git_dir = os.path.join(project_path, ".git")
            if not os.path.exists(git_dir):
                return {
                    'initialized': False,
                    'has_remote': False
                }
            
            # Get remote URL
            try:
                remote_result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                remote_url = remote_result.stdout.strip()
                has_remote = True
            except subprocess.CalledProcessError:
                remote_url = None
                has_remote = False
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=project_path,
                capture_output=True,
                text=True
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
            
            # Check for uncommitted changes
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_path,
                capture_output=True,
                text=True
            )
            has_changes = bool(status_result.stdout.strip())
            
            return {
                'initialized': True,
                'has_remote': has_remote,
                'remote_url': remote_url,
                'current_branch': current_branch,
                'has_uncommitted_changes': has_changes
            }
            
        except Exception as e:
            return {
                'initialized': False,
                'error': str(e)
            }
