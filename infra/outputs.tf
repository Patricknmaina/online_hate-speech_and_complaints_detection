# Terraform Outputs for AWS EC2 instance
output "public_ip" {
  description = "Public IP of the Safarimeter backend"
  value       = aws_instance.safarimeter_backend.public_ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_instance.safarimeter_backend.public_ip}"
}

output "api_url" {
  description = "FastAPI backend URL"
  value       = "http://${aws_instance.safarimeter_backend.public_ip}:8000"
}
