# Main Terraform configuration for AWS EC2 instance setup

resource "aws_key_pair" "safarimeter_key" {
  key_name   = "${var.project_name}-key"
  public_key = file("~/.ssh/id_rsa.pub")
}

resource "aws_security_group" "safarimeter_sg" {
  name        = "${var.project_name}-sg"
  description = "Security group for Safarimeter backend"

  # SSH access (restricted to your IP)
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_ip]
  }

  # FastAPI direct access
  ingress {
    description = "FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP (Nginx reverse proxy)
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS (future SSL)
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all outbound traffic
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

resource "aws_instance" "safarimeter_backend" {
  ami           = "ami-0fc5d935ebf8bc3bc" # Ubuntu 22.04 LTS (us-east-1)
  instance_type = var.instance_type
  key_name      = aws_key_pair.safarimeter_key.key_name

  vpc_security_group_ids = [
    aws_security_group.safarimeter_sg.id
  ]

  user_data = templatefile("${path.module}/user_data.sh", {
    github_repo_url = var.github_repo_url
    openai_api_key  = var.openai_api_key
  })

  tags = {
    Name = "${var.project_name}-backend"
  }
}
