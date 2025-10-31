#!/usr/bin/env python3
"""Quick test to verify AWS credentials are loaded correctly."""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 50)
print("AWS Credentials Check")
print("=" * 50)

# Check environment variables
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
bucket = os.getenv("UAE_S3_BUCKET", "akio-report-data")

print(f"\n✅ AWS_ACCESS_KEY_ID: {'Set' if access_key else '❌ NOT SET'}")
if access_key:
    print(f"   Value: {access_key[:10]}...{access_key[-4:]}")
    
print(f"\n✅ AWS_SECRET_ACCESS_KEY: {'Set' if secret_key else '❌ NOT SET'}")
if secret_key:
    print(f"   Value: {'*' * len(secret_key[:10])}...{secret_key[-4:]}")

print(f"\n✅ AWS_DEFAULT_REGION: {region}")
print(f"✅ UAE_S3_BUCKET: {bucket}")

if access_key and secret_key:
    print("\n" + "=" * 50)
    print("Testing AWS Connection...")
    print("=" * 50)
    
    try:
        import boto3
        s3 = boto3.client("s3", region_name=region)
        
        # Try to list buckets (tests credentials)
        response = s3.list_buckets()
        print("\n✅ AWS Connection Successful!")
        print(f"   Found {len(response['Buckets'])} bucket(s)")
        
        # Check if target bucket exists
        bucket_names = [b['Name'] for b in response['Buckets']]
        if bucket in bucket_names:
            print(f"\n✅ Target bucket '{bucket}' EXISTS")
        else:
            print(f"\n❌ Target bucket '{bucket}' NOT FOUND")
            print(f"   Available buckets: {', '.join(bucket_names)}")
            
        # Try a test upload (to verify write permissions)
        print("\n" + "=" * 50)
        print("Testing S3 Upload Permissions...")
        print("=" * 50)
        
        try:
            # Create a small test file
            import tempfile
            test_content = b"test file content"
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
                tmp.write(test_content)
                tmp_path = tmp.name
            
            test_key = f"uae_legislation/_test_upload.txt"
            s3.upload_file(tmp_path, bucket, test_key)
            print(f"✅ Upload test successful!")
            print(f"   Uploaded to: s3://{bucket}/{test_key}")
            
            # Clean up test file
            s3.delete_object(Bucket=bucket, Key=test_key)
            os.unlink(tmp_path)
            print(f"✅ Test file cleaned up")
            
        except Exception as upload_err:
            print(f"\n❌ Upload test FAILED: {upload_err}")
            print("\nPossible issues:")
            print("  - IAM user doesn't have s3:PutObject permission")
            print("  - Bucket policy doesn't allow uploads")
            print("  - Check IAM permissions for this user")
        
    except Exception as e:
        error_msg = str(e)
        if "InvalidAccessKeyId" in error_msg:
            print("\n❌ Invalid Access Key ID!")
            print("   The AWS Access Key ID you provided does not exist in AWS records.")
            print("   Please verify your credentials in .env file")
            print("   Go to AWS Console -> IAM -> Users -> Security Credentials to create new keys")
        elif "SignatureDoesNotMatch" in error_msg:
            print("\n❌ Invalid Secret Access Key!")
            print("   The AWS Secret Access Key doesn't match the Access Key ID.")
            print("   Please verify your credentials in .env file")
        elif "NoCredentialsError" in str(type(e).__name__):
            print("\n❌ No credentials found!")
            print("   boto3 couldn't find AWS credentials.")
            print("   Make sure .env file is loaded or set environment variables")
        else:
            print(f"\n❌ AWS Connection Failed: {e}")
else:
    print("\n❌ Credentials not found in environment!")
    print("\nTo fix:")
    print("  1. Make sure .env file exists with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    print("  2. Make sure python-dotenv is installed: pip install python-dotenv")
    print("  3. Make sure load_dotenv() is called in your code")

print("\n" + "=" * 50)
