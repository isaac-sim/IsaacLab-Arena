# prefix for created resources and tags
# full name looks like <prefix>.<deployment_name>.<app_name>.<resource_type>
variable "prefix" {
  default = "isaaclabarena-cloudci"
  type    = string
}

variable "deployment_name" {
  type = string
}

variable "region" {
  type = string
}

variable "aws_access_key_id" {
  type = string
}

variable "aws_secret_access_key" {
  type = string
}

variable "aws_session_token" {
  type    = string
  default = ""
}

variable "isaacsim_runner_enabled" {
  type = bool
}

variable "isaacsim_runner_instance_type" {
  type = string
}

variable "ssh_port" {
  default = 22
  type = string
}

variable "ingress_cidrs" {
  type = list(string)
}
