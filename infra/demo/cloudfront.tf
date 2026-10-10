# Public HTTPS in front of the demo copy (ADR-0047): a CloudFront distribution whose only way in is a VPC origin,
# a CloudFront-managed network interface inside our VPC that reaches the instance's private address. The instance
# keeps no public inbound path; its security group admits port 80 from CloudFront's service security group only.

resource "aws_cloudfront_vpc_origin" "demo" {
  vpc_origin_endpoint_config {
    name                   = "xcrs-demo"
    arn                    = aws_instance.demo.arn
    http_port              = 80
    https_port             = 443
    origin_protocol_policy = "http-only" # inside the VPC; visitors get HTTPS at the edge

    origin_ssl_protocols {
      items    = ["TLSv1.2"]
      quantity = 1
    }
  }

  timeouts {
    create = "30m" # a VPC origin takes ~15 minutes to deploy
    update = "30m"
    delete = "30m"
  }
}

# Created by CloudFront together with the VPC origin, in our VPC; AWS manages it, we only reference it.
data "aws_security_group" "cloudfront_vpc_origins" {
  vpc_id = aws_vpc.demo.id
  filter {
    name   = "group-name"
    values = ["CloudFront-VPCOrigins-Service-SG*"]
  }
  depends_on = [aws_cloudfront_vpc_origin.demo]
}

# The instance's only inbound rule: CloudFront (our distributions only) to Caddy on port 80.
resource "aws_vpc_security_group_ingress_rule" "cloudfront_http" {
  security_group_id            = aws_security_group.instance.id
  description                  = "CloudFront VPC origin to Caddy"
  ip_protocol                  = "tcp"
  from_port                    = 80
  to_port                      = 80
  referenced_security_group_id = data.aws_security_group.cloudfront_vpc_origins.id
}

# AWS-managed policies, looked up by name rather than hard-coded ids.
data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_cache_policy" "optimized" {
  name = "Managed-CachingOptimized"
}

data "aws_cloudfront_origin_request_policy" "all_viewer_except_host" {
  name = "Managed-AllViewerExceptHostHeader"
}

resource "aws_cloudfront_distribution" "demo" {
  enabled         = true
  comment         = "XCRS demo copy (ADR-0042, ADR-0047)"
  price_class     = "PriceClass_100" # North America and Europe edges: the cheapest class
  http_version    = "http2and3"
  is_ipv6_enabled = true

  origin {
    origin_id   = "xcrs-demo-instance"
    domain_name = aws_instance.demo.private_dns

    vpc_origin_config {
      vpc_origin_id            = aws_cloudfront_vpc_origin.demo.id
      origin_read_timeout      = 60 # free-text matching asks the LLM on VM-B; the most CloudFront allows by default
      origin_keepalive_timeout = 5
    }
  }

  # Pages and /api: never cached; all methods (the app POSTs); everything the visitor sent except Host.
  default_cache_behavior {
    target_origin_id         = "xcrs-demo-instance"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer_except_host.id
    compress                 = true
  }

  # Nuxt's build files carry a content hash in their names, so a cached copy can never be stale.
  ordered_cache_behavior {
    path_pattern           = "/_nuxt/*"
    target_origin_id       = "xcrs-demo-instance"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    cache_policy_id        = data.aws_cloudfront_cache_policy.optimized.id
    compress               = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  # The free *.cloudfront.net certificate until there is a domain (ADR-0040, ADR-0047).
  viewer_certificate {
    cloudfront_default_certificate = true
  }

  depends_on = [aws_vpc_security_group_ingress_rule.cloudfront_http]
}
