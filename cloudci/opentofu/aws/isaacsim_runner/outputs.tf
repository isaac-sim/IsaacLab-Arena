
output "public_ip" {
  value = aws_eip.eip.public_ip
}

output "vm_id" {
  value = aws_instance.instance.id
}
