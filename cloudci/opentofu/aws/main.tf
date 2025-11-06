terraform {
  required_version = ">= 1.3.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.41"
    }
  }
}

provider "aws" {
  region = var.region

  access_key = var.aws_access_key_id
  secret_key = var.aws_secret_access_key
  token      = var.aws_session_token

  default_tags {
    tags = {
      Deployment = "${var.deployment_name}"
    }
  }
}

module "common" {
  source = "./common"
  prefix = "${var.prefix}.${var.deployment_name}"
  region = var.region
}

module "isaacsim_runner" {
  source            = "./isaacsim_runner"
  prefix            = "${var.prefix}.${var.deployment_name}.isaacsim_runner"
  count             = var.isaacsim_runner_enabled ? 1 : 0
  keypair_id        = module.common.aws_key_pair_id
  instance_type     = var.isaacsim_runner_instance_type
  region            = var.region
  ssh_port          = var.ssh_port
  deployment_name   = var.deployment_name
  ingress_cidrs     = var.ingress_cidrs

  iam_instance_profile = null

  vpc = {
    id         = module.common.vpc.id
    cidr_block = module.common.vpc.cidr_block
  }
}
