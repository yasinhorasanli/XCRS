# What the instance itself may do in AWS: be reached by SSM Session Manager, and read the Tailscale key.

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "xcrs-demo-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Only this one parameter. It is encrypted with the AWS-managed aws/ssm key, which SSM may use on the
# caller's behalf, so no kms:Decrypt grant is needed.
data "aws_iam_policy_document" "read_tailscale_key" {
  statement {
    actions   = ["ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${var.tailscale_auth_key_parameter}"]
  }
}

resource "aws_iam_role_policy" "read_tailscale_key" {
  name   = "read-tailscale-auth-key"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.read_tailscale_key.json
}

resource "aws_iam_instance_profile" "instance" {
  name = "xcrs-demo-instance"
  role = aws_iam_role.instance.name
}
