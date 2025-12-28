
output "ssh_key" {
  value     = module.common.ssh_key.private_key_pem
  sensitive = true
}

output "cloud" {
  value = "aws"
}

output "isaacsim_runner_ip" {
  value = var.isaacsim_runner_enabled ? module.isaacsim_runner[0].public_ip : "NA"
}

output "isaacsim_runner_vm_id" {
  value = try(var.isaacsim_runner_enabled ? module.isaacsim_runner[0].vm_id : "NA", "NA")
}

