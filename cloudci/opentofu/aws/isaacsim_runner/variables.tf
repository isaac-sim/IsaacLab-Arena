variable "prefix" {
  type = string
}

variable "keypair_id" {
  type = string
}

variable "instance_type" {
  type = string
}

variable "region" {
  type = string
}

variable "from_image" {
  default = true
  type    = bool
}

variable "vpc" {
  type = object({
    id         = string,
    cidr_block = string,
  })
}

variable "iam_instance_profile" {
  default = null
  type    = string
}

variable "deployment_name" {
  type = string
}

# base - used when from_image is *false*
variable "base_ami_name" {
  default = "ubuntu/images/hvm-ssd/ubuntu-*-22.04-amd64-server-*"
}

variable "ssh_port" {
  type = number
}

# for general use, ["0.0.0.0/0"] is ok
# but may be helpful for accounts with stricter security policies
variable "ingress_cidrs" {
  type = list(string)
}
