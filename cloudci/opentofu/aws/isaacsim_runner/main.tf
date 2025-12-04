# query availability zones where
# we can launch instances of type required

data "aws_ec2_instance_type_offerings" "zones" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
  location_type = "availability-zone"
}

# create a subnet for the isaac instance

resource "aws_subnet" "subnet" {
  # get a /24 block from vpc cidr
  cidr_block              = cidrsubnet(var.vpc.cidr_block, 8, 3)
  availability_zone       = try(sort(data.aws_ec2_instance_type_offerings.zones.locations)[0], "not-available")
  vpc_id                  = var.vpc.id
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.prefix}.subnet"
  }
}

# instance
resource "aws_instance" "instance" {
  ami                    = data.aws_ami.ami.id
  instance_type          = var.instance_type
  key_name               = var.keypair_id
  vpc_security_group_ids = [aws_security_group.sg.id]
  subnet_id              = aws_subnet.subnet.id
  iam_instance_profile   = var.iam_instance_profile

  root_block_device {
    volume_type           = "gp3"
    volume_size           = "256" # GB
    delete_on_termination = true

    tags = {
      Name       = "${var.prefix}.root_ebs"
      Deployment = "${var.deployment_name}"
    }
  }

  tags = {
    Name = "${var.prefix}.vm"
  }
}

# elastic ip
resource "aws_eip" "eip" {
  instance = aws_instance.instance.id
  tags = {
    Name = "${var.prefix}.eip"
  }
}
