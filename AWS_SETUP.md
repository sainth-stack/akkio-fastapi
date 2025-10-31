# AWS Credentials Setup Guide

## Quick Setup (Choose One Method)

### Method 1: Environment Variables (Recommended)

**Add to your shell profile (~/.zshrc or ~/.bashrc):**
```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1
```

**Or set temporarily in terminal:**
```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1
```

**After setting, reload shell:**
```bash
source ~/.zshrc  # or source ~/.bashrc
```

### Method 2: AWS Credentials File (Most Reliable)

**Create credentials file:**
```bash
mkdir -p ~/.aws
nano ~/.aws/credentials
```

**Add this content:**
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
region = us-east-1
```

**Set permissions:**
```bash
chmod 600 ~/.aws/credentials
```

### Method 3: Use Setup Script

Run the automated setup script:
```bash
./setup_aws_credentials.sh
```

---

## How to Get AWS Credentials

1. **Go to AWS Console**: https://console.aws.amazon.com/
2. **Navigate to IAM**: Click on "IAM" in services
3. **Go to Users**: Click "Users" in left sidebar
4. **Select Your User**: Click on your username
5. **Security Credentials Tab**: Click "Security credentials"
6. **Create Access Key**: 
   - Click "Create access key"
   - Choose "Application running outside AWS"
   - Click "Next"
   - (Optional) Add description tag
   - Click "Create access key"
7. **Copy Credentials**:
   - Copy the Access Key ID
   - Copy the Secret Access Key (shown only once!)
   - **Important**: Save the Secret Access Key immediately!

---

## Verify Installation

**Test if credentials work:**
```bash
# Activate venv
source venv/bin/activate

# Test AWS access
python3 -c "import boto3; s3 = boto3.client('s3'); print('✅ AWS credentials work!'); print('Buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])"
```

---

## Verify S3 Bucket

Make sure your bucket exists and has correct name:
```bash
# Your bucket name should be: akio-report-data
# Check bucket exists:
python3 -c "import boto3; s3 = boto3.client('s3'); buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]; print('Your buckets:', buckets); print('akio-report-data' in buckets and '✅ Bucket exists!' or '❌ Bucket not found!')"
```

---

## S3 Permissions Required

Your IAM user needs these S3 permissions:
- `s3:PutObject` - to upload files
- `s3:GetObject` - to read files (optional)
- `s3:ListBucket` - to list bucket contents (optional)

**Example IAM Policy:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::akio-report-data/uae_legislation/*"
        }
    ]
}
```

---

## Troubleshooting

### "NoCredentialsError"
- Credentials not found in environment or ~/.aws/credentials
- **Fix**: Set credentials using one of the methods above

### "AccessDenied"
- IAM user doesn't have S3 permissions
- **Fix**: Add S3 permissions to IAM user policy

### "NoSuchBucket"
- Bucket name doesn't exist or wrong
- **Fix**: Check bucket name in `.env` or environment variable `UAE_S3_BUCKET`

---

## Quick Commands Summary

```bash
# Install boto3 (already installed ✅)
source venv/bin/activate
pip install boto3

# Set environment variables (current session)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Or use credentials file
mkdir -p ~/.aws
nano ~/.aws/credentials  # Add credentials as shown above
chmod 600 ~/.aws/credentials

# Test AWS access
python3 -c "import boto3; s3 = boto3.client('s3'); print(s3.list_buckets())"
```




