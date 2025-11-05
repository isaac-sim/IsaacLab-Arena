# Proposal: Cloud-Based CI for IsaacLab Arena

## Motivation

Upon looking at the current CI setup and challenges, it seems that adding a cloud-based CI, could help with:

- Faster turnaround times on MRs due to ability to scale up and down as needed
- Testing on a variety of GPUs, including ones, users are likely to run on (Isaac Lab is often used in the cloud) - A10G, L4, L40, T4, etc.
- Cost effectiveness (more on that below)

Initially, it can be implemented on AWS (A10G, L4, L40S, T4, H100, B100, GB200), and later expanded to GCP (T4, L4, RTX PRO 6000 - GA soon, B200, GB200) and Azure (A10, H100, TBD) as needed.

## Cost effectiveness analysis

Here is a table outlining hour of use on various GPU instances (on-demand, single GPU, 256GB permanent sortage):

| Cloud Provider | GPU Model | Instance Type | Approx. Hourly Cost  |
| ---  | --- | --- | ---  |
| AWS  | T4  | g4dn.xlarge | ~$0.6 |
| AWS  | A10G  | g5.xlarge | ~$1.3 |
| AWS  | L4  | g6.xlarge | ~$1  |
| AWS  | L40S  | g6e.xlarge  | ~$2  |
| AWS | H100  | p5.4xlarge | ~$7 |
| AWS | B100  |  TBD  | TBD  |
| AWS | GB200  |  TBD  | TBD  |

(to be continued with GCP and Azure data)

Those costs are charged only if the instance is running. If it is stopped, only the minimal storage and IP address costs are incurred. This means that if the CI jobs are not run for some time, the costs are minimal as instances an be automatically stopped.

## Architecture

Cloud CI setup will be very loosely based on the Isaac Automator ([https://github.com/NVIDIA-Omniverse/IsaacAutomator](https://github.com/isaac-sim/IsaacAutomator)), but will be simplified to avoid the need to support any unneeded functionality.

There will be 3 main components:

1. CLI tool to:
 - Deploy an instance
 - Connect to the instance
 - Start, stop an instance
 - Destroy an instance and all associated resources
2. Terraform scripts for clouds that are implemented, starting with AWS
3. Ansible tasks to configure the instance
4. Optionally packer to be able to save AMIs for faster startup times and consistent environments.

## Security considerations

Isaac Arena is open source, so there should not be any concerns with running upcoming changes in the cloud, especially if the access is properly secured.

Access conbtrol can be implemented by:

- Limiting range of IP addresses that can connect to the instances
- Setting up reliable key-based authentication for every new instance
- Using least-privilege IAM (or equivalents) roles

## Timeline

Since this can be based on existing Isaac Automator code, the timeline can be relatively short:

| Step | Duration | Outcome  |
|--------------------|----------|-----------------------------------------|
| Prototype  | 2 weeks  | Cloud CI workflow working "somehow"  |
| Testing & Polishing| 2 weeks  | Other members of Isaac Arena team are happy with it |
| Final Integration  | 1 week | MR is merged and CI is run (as needed or regularly) using the cloud CI tools |
