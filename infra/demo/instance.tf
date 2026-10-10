# The demo copy: one Graviton instance running the compose stack (ADR-0042), set up on first boot by cloud-init.

data "aws_ssm_parameter" "al2023_arm64" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

resource "aws_instance" "demo" {
  ami                    = nonsensitive(data.aws_ssm_parameter.al2023_arm64.value) # a public AMI id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.instance.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  # "standard" throttles the CPU when its burst credits run out instead of billing for extra credits
  # ("unlimited", the T4g default, can cost money even inside the free trial).
  credit_specification {
    cpu_credits = "standard"
  }

  # IMDSv2 only, and one network hop: the host's AWS CLI can read the instance role, containers can't.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true # disposable: the catalog and embeddings are rebuilt on every first boot
  }

  user_data = templatefile("${path.module}/cloud-init.sh.tftpl", {
    region              = var.region
    release             = var.release
    tailscale_key_param = var.tailscale_auth_key_parameter
    tailnet_hostname    = var.tailnet_hostname
    ollama_url          = var.vm_b_ollama_url
    compose_version     = var.compose_version
    compose_sha256      = var.compose_sha256
  })
  user_data_replace_on_change = true # a changed first-boot script means a new instance

  lifecycle {
    # A newer Amazon Linux image shouldn't replace the instance on an unrelated apply;
    # `terraform apply -replace=aws_instance.demo` picks it up on purpose.
    ignore_changes = [ami]
  }

  tags = { Name = "xcrs-demo" }
}
