# A VPC of our own with one public subnet and no NAT gateway (~$35/month): the instance reaches the internet
# (GHCR, package repos, Tailscale, SSM) through its own public IPv4 address. Nothing can connect in.

resource "aws_vpc" "demo" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "xcrs-demo" }
}

resource "aws_internet_gateway" "demo" {
  vpc_id = aws_vpc.demo.id
  tags   = { Name = "xcrs-demo" }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.demo.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true # needed for outbound traffic without a NAT gateway; GHCR has no IPv6
  tags                    = { Name = "xcrs-demo-public" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.demo.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.demo.id
  }
  tags = { Name = "xcrs-demo-public" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# Every VPC comes with a default security group that lets its members talk to each other; take its rules away
# so nothing uses it by accident.
resource "aws_default_security_group" "demo" {
  vpc_id = aws_vpc.demo.id
  tags   = { Name = "xcrs-demo-default-unused" }
}

# No inbound rules at all: shells go through SSM Session Manager, the site through Tailscale (and CloudFront
# later). Security groups are stateful, so replies to the instance's own outbound connections still arrive.
resource "aws_security_group" "instance" {
  name        = "xcrs-demo-instance"
  description = "XCRS demo: no inbound, all outbound"
  vpc_id      = aws_vpc.demo.id
  tags        = { Name = "xcrs-demo-instance" }
}

resource "aws_vpc_security_group_egress_rule" "all_ipv4" {
  security_group_id = aws_security_group.instance.id
  description       = "Outbound: images, packages, Tailscale, SSM"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}
