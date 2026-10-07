output "instance_id" {
  value = aws_instance.demo.id
}

output "shell" {
  description = "A shell on the instance without SSH (needs the Session Manager plugin)."
  value       = "aws ssm start-session --profile xcrs --target ${aws_instance.demo.id}"
}

output "first_boot_log" {
  value = "sudo tail -f /var/log/cloud-init-output.log"
}

output "site" {
  description = "Tailnet only until CloudFront (step 5)."
  value       = "https://${var.tailnet_hostname}.tail3afc6e.ts.net"
}
