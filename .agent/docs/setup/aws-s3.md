# AWS S3 Configuration for DVC - Complete Guide

## Overview

This guide documents the AWS S3 setup for the TFG Market Intelligence project. The configuration enables **dual storage**: MinIO (local, default) and AWS S3 (cloud, optional).

---

## Configuration Summary

### Bucket Details

- **Bucket name**: `tfg-market-intelligence-jmedrano`
- **Region**: `eu-west-1` (Ireland)
- **Encryption**: SSE-S3 (server-side, AWS managed keys)
- **Public access**: BLOCKED (all settings)
- **Versioning**: Disabled (DVC handles versioning)

### IAM User

- **Username**: `dvc-tfg-user`
- **Permissions**: `AmazonS3FullAccess` (S3 only, no other AWS services)
- **Access type**: Programmatic (CLI/API only, no console access)

### DVC Remotes

```bash
storage  s3://tfg-market-intelligence              # MinIO local (default)
aws-s3   s3://tfg-market-intelligence-jmedrano/dvc-storage  # AWS S3
```

### Current Status

- ✅ **11 files** pushed successfully (~550 MB)
- ✅ Billing alerts configured (Free Tier + Zero Spend Budget)
- ✅ AWS CLI configured with credentials
- ✅ Connection verified

---

## Cost Safety

### Free Tier Limits (12 months)

- **Storage**: 5 GB/month
- **PUT requests**: 2,000/month
- **GET requests**: 20,000/month

### Project Usage Estimate

- **Current size**: ~550 MB (11% of quota)
- **Files**: 11 files
- **Estimated PUT/month**: ~15-30 (1.5% of quota)
- **Safety margin**: ~100 full pushes/month possible

### Alerts Configured

1. **Free Tier Usage Alerts**: Email when approaching limits
2. **Zero Spend Budget**: Email if any charge > $0.01 occurs

> **✅ CONFIRMED**: No costs will be incurred if you stay within Free Tier limits.

---

## Usage Commands

### Push Data to S3

```bash
# Activate virtual environment first
source .venv/bin/activate

# Push to AWS S3 (explicit remote)
dvc push -r aws-s3

# Push to default remote (MinIO local)
dvc push
```

### Pull Data from S3

```bash
# Pull from AWS S3
dvc pull -r aws-s3

# Pull from default remote (MinIO)
dvc pull
```

### List Remote Files

```bash
# Via AWS CLI
aws s3 ls s3://tfg-market-intelligence-jmedrano/dvc-storage/ --recursive

# Via DVC
dvc list -R --dvc-only -r aws-s3 .
```

### Check DVC Remotes

```bash
# List all remotes
dvc remote list

# Show specific remote config
dvc remote list --show-origin
```

---

## Switching Default Remote (Optional)

If you want to use AWS S3 as default instead of MinIO:

```bash
# Set aws-s3 as default
dvc remote default aws-s3

# Verify
dvc remote list
```

To revert:

```bash
# Set storage (MinIO) as default again
dvc remote default storage
```

---

## Security Best Practices

### ✅ Implemented

- [x] Public access blocked on bucket
- [x] IAM user with minimal permissions (S3 only)
- [x] Credentials stored in `~/.aws/credentials` (not in repo)
- [x] Billing alerts configured
- [x] Encryption enabled (SSE-S3)

### ⚠️ Recommendations

- **DO NOT** commit AWS credentials to Git
- **DO NOT** share Access Key ID / Secret Access Key publicly
- **DO NOT** enable other AWS services (EC2, Lambda, etc.) to avoid costs
- **VERIFY** monthly usage in AWS Billing Dashboard

---

## Configuration Files

### `.dvc/config`

```ini
[core]
    remote = storage

['remote "storage"']
    url = s3://tfg-market-intelligence
    endpointurl = http://localhost:9000
    access_key_id = minioadmin
    secret_access_key = minioadmin
    use_ssl = false

['remote "aws-s3"']
    url = s3://tfg-market-intelligence-jmedrano/dvc-storage
    region = eu-west-1
```

### `~/.aws/credentials`

```ini
[default]
aws_access_key_id = AKIA...REDACTED
aws_secret_access_key = ***REDACTED***
```

### `~/.aws/config`

```ini
[default]
region = eu-west-1
output = json
```

---

## Troubleshooting

### Error: "Access Denied"

**Solution**: Verify IAM user permissions

```bash
aws iam list-attached-user-policies --user-name dvc-tfg-user
```

### Error: "Bucket does not exist"

**Solution**: Verify bucket name and region

```bash
aws s3 ls | grep tfg-market-intelligence
```

### Error: "dvc command not found"

**Solution**: Activate virtual environment

```bash
source .venv/bin/activate
dvc --version
```

### Verify AWS CLI Configuration

```bash
aws configure list
aws sts get-caller-identity  # Shows current IAM user
```

---

## Monthly Monitoring

### Check AWS Usage

1. Go to **AWS Console** → **Billing Dashboard**
2. Check **Free Tier Usage**
3. Verify S3 metrics:
   - Storage: Should be < 5 GB
   - PUT requests: Should be < 2,000/month
   - GET requests: Should be < 20,000/month

### Check DVC Usage

```bash
# See remote storage usage
dvc status -r aws-s3

# List all files in remote
aws s3 ls s3://tfg-market-intelligence-jmedrano/dvc-storage/ --recursive --summarize
```

---

## Migration Path

### When to Use Each Remote

| Scenario          | Recommended Remote |
| ----------------- | ------------------ |
| Local development | `storage` (MinIO)  |
| Backup to cloud   | `aws-s3` (AWS S3)  |
| Collaboration     | `aws-s3` (AWS S3)  |
| Remote demo       | `aws-s3` (AWS S3)  |
| CI/CD pipeline    | `aws-s3` (AWS S3)  |

### Future: Full Migration to S3

If you decide to migrate fully to S3:

```bash
# Set aws-s3 as default
dvc remote default aws-s3

# Update .dvc/config in git
git add .dvc/config
git commit -m "chore: migrate DVC remote from MinIO to AWS S3"

# Push to new default
dvc push
```

---
