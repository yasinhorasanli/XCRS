# The bucket that holds Terraform state for every other stack (ADR-0042). Created once, with local state;
# afterwards this stack's own state moves into the bucket too (backend.tf, see infra/README.md).

variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "state_bucket" {
  description = "Globally unique; a random suffix instead of the account id, so the name can be public."
  type        = string
  default     = "xcrs-tfstate-d6b0fd"
}

resource "aws_s3_bucket" "state" {
  bucket = var.state_bucket

  # Losing the state means Terraform forgets what it created; never delete this bucket by accident.
  lifecycle {
    prevent_destroy = true
  }
}

# Every write keeps the previous version, so a corrupted or wrongly applied state can be rolled back.
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

# SSE-S3 (free); KMS keys would cost $1/month each and add nothing for a single-person account.
resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ACLs off: access is decided by IAM and the bucket policy only.
resource "aws_s3_bucket_ownership_controls" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# State can hold secrets (e.g. generated passwords): refuse anything that isn't HTTPS.
data "aws_iam_policy_document" "state" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.state.arn,
      "${aws_s3_bucket.state.arn}/*",
    ]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "state" {
  bucket = aws_s3_bucket.state.id
  policy = data.aws_iam_policy_document.state.json

  depends_on = [aws_s3_bucket_public_access_block.state]
}

# Old state versions are only for rollback: one is deleted 90 days after it was replaced, but the 10 newest
# old versions are always kept.
resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    id     = "expire-old-state-versions"
    status = "Enabled"
    filter {}

    noncurrent_version_expiration {
      noncurrent_days           = 90
      newer_noncurrent_versions = 10
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [aws_s3_bucket_versioning.state]
}

output "state_bucket" {
  value = aws_s3_bucket.state.id
}

output "region" {
  value = var.region
}
