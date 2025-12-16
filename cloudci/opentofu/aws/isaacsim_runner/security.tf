# security group for isaac
resource "aws_security_group" "sg" {
  name   = "${var.prefix}.sg"
  vpc_id = var.vpc.id

  tags = {
    Name = "${var.prefix}.sg"
  }

  # ssh
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ingress_cidrs
  }

  # nomachine
  ingress {
    from_port   = 4000
    to_port     = 4000
    protocol    = "tcp"
    cidr_blocks = var.ingress_cidrs
  }
  ingress {
    from_port   = 4000
    to_port     = 4000
    protocol    = "udp"
    cidr_blocks = var.ingress_cidrs
  }

  # vnc
  ingress {
    from_port   = 5900
    to_port     = 5900
    protocol    = "tcp"
    cidr_blocks = var.ingress_cidrs
  }

  # novnc
  ingress {
    from_port   = 6080
    to_port     = 6080
    protocol    = "tcp"
    cidr_blocks = var.ingress_cidrs
  }

  # allow outbound traffic

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

# custom ssh port
resource "aws_security_group_rule" "custom_ssh" {
  count             = var.ssh_port != 22 ? 1 : 0
  type              = "ingress"
  security_group_id = aws_security_group.sg.id
  from_port         = var.ssh_port
  to_port           = var.ssh_port
  protocol          = "tcp"
  cidr_blocks       = var.ingress_cidrs
}
