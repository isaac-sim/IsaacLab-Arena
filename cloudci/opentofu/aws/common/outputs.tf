output "ssh_key" {
  value = tls_private_key.ssh_key
}

output "aws_key_pair_id" {
  value = aws_key_pair.keypair.id
}

output "vpc" {
  value = {
    id         = aws_vpc.vpc.id
    cidr_block = aws_vpc.vpc.cidr_block
  }
}
