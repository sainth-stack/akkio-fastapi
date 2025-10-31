#!/bin/bash

# AWS Credentials Setup Script for UAE Legislation Ingestion
# This script helps you configure AWS credentials for S3 uploads

echo "=========================================="
echo "AWS Credentials Setup for S3 Uploads"
echo "=========================================="
echo ""
echo "You need AWS credentials to upload PDFs to S3."
echo ""
echo "How to get AWS credentials:"
echo "1. Go to AWS Console: https://console.aws.amazon.com/"
echo "2. Navigate to: IAM -> Users -> Your User -> Security Credentials"
echo "3. Click 'Create access key'"
echo "4. Choose 'Application running outside AWS'"
echo "5. Copy the Access Key ID and Secret Access Key"
echo ""
echo "Setup Options:"
echo "1) Set environment variables in current shell"
echo "2) Create .env file (requires python-dotenv)"
echo "3) Configure AWS credentials file (~/.aws/credentials)"
echo ""
read -p "Choose option (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "Enter your AWS credentials:"
        read -p "AWS_ACCESS_KEY_ID: " access_key
        read -p "AWS_SECRET_ACCESS_KEY: " secret_key
        read -p "AWS_DEFAULT_REGION (default: us-east-1): " region
        region=${region:-us-east-1}
        
        echo ""
        echo "Adding to your shell profile..."
        echo ""
        echo "# AWS Credentials for S3 Uploads" >> ~/.zshrc
        echo "export AWS_ACCESS_KEY_ID=$access_key" >> ~/.zshrc
        echo "export AWS_SECRET_ACCESS_KEY=$secret_key" >> ~/.zshrc
        echo "export AWS_DEFAULT_REGION=$region" >> ~/.zshrc
        echo ""
        echo "✅ Credentials added to ~/.zshrc"
        echo "Run: source ~/.zshrc  (or restart terminal)"
        ;;
    2)
        if [ ! -f .env ]; then
            cp .env.example .env
        fi
        echo ""
        echo "Edit .env file with your credentials:"
        echo "AWS_ACCESS_KEY_ID=your_access_key"
        echo "AWS_SECRET_ACCESS_KEY=your_secret_key"
        echo "AWS_DEFAULT_REGION=us-east-1"
        echo ""
        echo "Then make sure your app loads .env file using python-dotenv:"
        echo "  from dotenv import load_dotenv"
        echo "  load_dotenv()"
        ;;
    3)
        echo ""
        mkdir -p ~/.aws
        read -p "AWS_ACCESS_KEY_ID: " access_key
        read -p "AWS_SECRET_ACCESS_KEY: " secret_key
        read -p "AWS_DEFAULT_REGION (default: us-east-1): " region
        region=${region:-us-east-1}
        
        cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $access_key
aws_secret_access_key = $secret_key
region = $region
EOF
        
        echo ""
        echo "✅ AWS credentials configured in ~/.aws/credentials"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Make sure your S3 bucket 'akio-report-data' exists"
echo "2. Ensure your IAM user has S3 write permissions"
echo "3. Restart your FastAPI server"
echo "=========================================="




