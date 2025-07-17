# Isaac Arena CI

We run a local gitlab CI runner in the Zurich office (rather than using an NVIDIA maintained machine).

The reasons for this are:
- So we can have a GPU accessible in CI
- We can have large datasets/models stored locally on the CI machine.

As things change we will need to:
- Update the test checkpoint.
- Update local data (if we have any).

To do so you will need to put these files on the CI machine directly.

## Log In to the CI machine

To log in to the CI machine do:

```bash
ssh gitlab-runner@shallowlearning.dyn.nvidia.com
```

_password: ASK ALEX_

## Data and Checkpoint Locations

Data is located at:

```bash
/home/gitlab-runner/datasets
```

which is mapped to `/datasets` in the launched test docker.

Checkpoints are located at:

```bash
/home/gitlab-runner/models
```

which is mapped to `/models` in the launched test docker.


## Gitlab Runner Configuration

We use this to affect how the test docker is launched. We so far have used this to:
- Mount the datasets and models
- Add the `graphics` capability to the nvidia-docker launch.

This is located at:

```bash
/etc/gitlab-runner/config.toml
```

After making modifications, run the following command to restart the gitlab runner docker:
```bash
sudo gitlab-runner restart
```

## Gitlab Runner Status

The runner is running as a systemd job. So you can go:

```bash
sudo systemctl status gitlab-runner.service
```
