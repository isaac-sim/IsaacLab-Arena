variable "prefix" {
  type = string
}

variable "ssh_key" {
  default = null
}

variable "region" {
  type = string
}

variable "vpc_cidr_block" {
  default = "10.1.0.0/16"
  type = string
}
