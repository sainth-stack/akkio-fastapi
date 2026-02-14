#!/usr/bin/env python3
"""
Check where generated project files are located
Run this on your FastAPI EC2 to verify paths
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_builder.services.runtime_paths import get_projects_dir

print("=" * 60)
print("Project Paths Verification")
print("=" * 60)

# Where files are written
projects_dir = get_projects_dir()
print(f"\n📁 Projects Directory (where code is saved):")
print(f"   {projects_dir}")

# Check if directory exists
if os.path.exists(projects_dir):
    print(f"   ✅ Directory exists")
    
    # List all projects
    try:
        projects = [d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))]
        if projects:
            print(f"\n📦 Found {len(projects)} project(s):")
            for proj in sorted(projects, reverse=True)[:10]:  # Show latest 10
                proj_path = os.path.join(projects_dir, proj)
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(proj_path)
                    for filename in filenames
                )
                print(f"   - {proj} ({size:,} bytes)")
        else:
            print(f"\n⚠️  No projects found in directory")
    except Exception as e:
        print(f"\n❌ Error listing projects: {e}")
else:
    print(f"   ❌ Directory does not exist!")
    print(f"   Create it with: mkdir -p {projects_dir}")

# Environment variables
print(f"\n🔧 Environment Variables:")
runtime_dir = os.getenv("APP_BUILDER_RUNTIME_DIR")
if runtime_dir:
    print(f"   APP_BUILDER_RUNTIME_DIR = {runtime_dir}")
else:
    print(f"   APP_BUILDER_RUNTIME_DIR = (not set, using default)")

projects_dir_env = os.getenv("APP_BUILDER_PROJECTS_DIR")
if projects_dir_env:
    print(f"   APP_BUILDER_PROJECTS_DIR = {projects_dir_env} (deprecated, not used)")

print("\n" + "=" * 60)
print("✅ Use this path in your deployment configuration")
print("=" * 60)
