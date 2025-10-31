#!/usr/bin/env python3
"""Test AWS credentials from .env file"""
import os
from dotenv import load_dotenv
import boto3

# Load .env file
load_dotenv()

# Get credentials
aws_key = os.getenv('AWS_ACCESS_KEY_ID', 'NOT_FOUND')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY', 'NOT_FOUND')
aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

print("=" * 50)
print("AWS Credentials Test")
print("=" * 50)
print(f"Access Key ID: {aws_key[:10]}... (length: {len(aws_key) if aws_key != 'NOT_FOUND' else 0})")
print(f"Secret Key: {'*' * 20 if aws_secret != 'NOT_FOUND' else 'NOT_FOUND'} (length: {len(aws_secret) if aws_secret != 'NOT_FOUND' else 0})")
print(f"Region: {aws_region}")
print()

if aws_key == 'NOT_FOUND' or aws_secret == 'NOT_FOUND':
    print("❌ Credentials not found in environment!")
    print("   Make sure .env file exists and has AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    exit(1)

# Verify Access Key ID format (should be 20 characters starting with AKIA)
if len(aws_key) != 20 or not aws_key.startswith('AKIA'):
    print("⚠️  Warning: Access Key ID format looks wrong")
    print(f"   Expected: 20 characters starting with 'AKIA'")
    print(f"   Got: {len(aws_key)} characters starting with '{aws_key[:4]}'")

# Test credentials with AWS
print("Testing AWS connection...")
try:
    s3 = boto3.client('s3', region_name=aws_region)
    response = s3.list_buckets()
    print("✅ AWS credentials are VALID!")
    print(f"✅ Connected to AWS successfully")
    print(f"✅ Buckets: {[b['Name'] for b in response['Buckets']]}")
    print()
    
    # Check if target bucket exists
    bucket_name = os.getenv('UAE_S3_BUCKET', 'akio-report-data')
    buckets = [b['Name'] for b in response['Buckets']]
    if bucket_name in buckets:
        print(f"✅ Target bucket '{bucket_name}' exists")
    else:
        print(f"⚠️  Target bucket '{bucket_name}' NOT FOUND")
        print(f"   Available buckets: {buckets}")
        
except Exception as e:
    error_type = type(e).__name__
    error_msg = str(e)
    print(f"❌ Error: {error_type}")
    print(f"   Message: {error_msg}")
    print()
    
    if "InvalidAccessKeyId" in error_msg or "InvalidAccessKeyId" in error_type:
        print("🔍 Troubleshooting InvalidAccessKeyId:")
        print("   1. Go to AWS Console: https://console.aws.amazon.com/iam/")
        print("   2. Navigate to: IAM -> Users -> Your User -> Security Credentials")
        print("   3. Check if your Access Key ID exists and is active")
        print("   4. If not, create a new access key")
        print("   5. Update .env file with the new Access Key ID and Secret Key")
        print()
        print(f"   Current Access Key ID (first 10 chars): {aws_key[:10]}...")
        print("   Make sure this matches exactly with AWS Console")
        
    elif "SignatureDoesNotMatch" in error_msg:
        print("🔍 Troubleshooting SignatureDoesNotMatch:")
        print("   1. Check if Secret Access Key is correct")
        print("   2. Make sure there are no extra spaces or quotes in .env file")
        print("   3. Secret key should NOT have quotes around it in .env")
        print()
        
    elif "AccessDenied" in error_msg or "Forbidden" in error_msg:
        print("🔍 Troubleshooting Access Denied:")
        print("   1. Your IAM user needs S3 permissions")
        print("   2. Check IAM policies for your user")
        print("   3. Ensure user has s3:PutObject permission")
        
    else:
        print("🔍 General troubleshooting:")
        print("   1. Check AWS credentials in .env file")
        print("   2. Verify AWS region is correct")
        print("   3. Check your internet connection")
        print("   4. Verify AWS service is available")

print("=" * 50)




