# Variables for AWS EC2 instance setup and configuration

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "allowed_ssh_ip" {
  description = "Your public IP for SSH access (CIDR format, e.g. 1.2.3.4/32)"
  type        = string
}

variable "project_name" {
  description = "Project name used for resource naming and tags"
  type        = string
  default     = "safarimeter"
}

variable "github_repo_url" {
  description = "GitHub repository URL to clone on the EC2 instance"
  type        = string
  default     = "https://github.com/patricknmaina/online_hate-speech_and_complaints_detection.git"
}

variable "openai_api_key" {
  description = "OpenAI API key for the FastAPI backend"
  type        = string
  sensitive   = true
  default     = ""
}
