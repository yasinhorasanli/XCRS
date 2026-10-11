# The demo's own name (ADR-0051): aws.xcrs.app → CloudFront. The certificate comes from ACM in us-east-1 (the only
# region CloudFront takes certificates from), proven by a DNS record in Cloudflare. Cloudflare's proxy stays off for
# these records: two CDNs in a row would interfere, and CloudFront already terminates TLS.

data "cloudflare_zone" "xcrs" {
  filter = { name = var.zone }
}

resource "aws_acm_certificate" "public" {
  region            = "us-east-1"
  domain_name       = var.public_hostname
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "cloudflare_dns_record" "acm_validation" {
  for_each = {
    for o in aws_acm_certificate.public.domain_validation_options : o.domain_name => {
      name    = trimsuffix(o.resource_record_name, ".")
      type    = o.resource_record_type
      content = trimsuffix(o.resource_record_value, ".")
    }
  }

  zone_id = data.cloudflare_zone.xcrs.id
  name    = each.value.name
  type    = each.value.type
  content = each.value.content
  proxied = false
  ttl     = 300
  comment = "ACM validation for ${each.key} (infra/demo)"
}

resource "aws_acm_certificate_validation" "public" {
  region                  = "us-east-1"
  certificate_arn         = aws_acm_certificate.public.arn
  validation_record_fqdns = [for r in cloudflare_dns_record.acm_validation : r.name]
}

resource "cloudflare_dns_record" "public" {
  zone_id = data.cloudflare_zone.xcrs.id
  name    = var.public_hostname
  type    = "CNAME"
  content = aws_cloudfront_distribution.demo.domain_name
  proxied = false
  ttl     = 300
  comment = "the AWS demo copy on CloudFront (infra/demo, ADR-0051)"
}

# Search engines should send people to xcrs.app, not to this copy.
resource "aws_cloudfront_response_headers_policy" "noindex" {
  name    = "xcrs-demo-noindex"
  comment = "Keep the AWS copy out of search results (ADR-0051)"

  custom_headers_config {
    items {
      header   = "X-Robots-Tag"
      value    = "noindex"
      override = true
    }
  }
}
