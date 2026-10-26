# Community Contributions

Thank you for helping improve IsaacLab-Arena.

We welcome bug reports, feature ideas, documentation improvements, and code contributions. Every contribution—large or small—helps make IsaacLab-Arena more useful for the community.

## Proposing a Change

For a new feature, public API change, or significant design change, please open a [Design Proposal](https://github.com/isaac-sim/IsaacLab-Arena/issues/new?template=design-proposal.yml) issue before submitting a full implementation.

This helps maintainers and contributors:

- Understand the problem and intended users.
- Discuss the design early.
- Avoid duplicated or conflicting work.
- Assess fit with the project direction and maintenance cost.

Please use the [Design Proposal form](https://github.com/isaac-sim/IsaacLab-Arena/issues/new?template=design-proposal.yml) and include:

- Motivation
- Goals and non-goals
- High-level API
- Implementation
- Gaps and open questions
- Implementation readiness

For smaller proposals, a proof of concept and alternatives considered are optional. Maintainers may request them when the proposal affects public APIs, multiple modules, performance, compatibility, or long-term maintenance.

## How We Prioritize

Maintainer bandwidth is limited, and proposals are evaluated alongside all other project work.

We generally prioritize contributions that have:

- Clear value for multiple IsaacLab-Arena users.
- Strong alignment with project goals.
- A focused, understandable design.
- Clear compatibility impact and a sustainable long-term maintenance plan.
- A contributor willing to help implement and maintain the change.
- A ready-to-review implementation that meets project quality standards.

An accepted issue or design proposal means the direction is welcome. It does not guarantee an implementation timeline or that maintainers will implement the feature.

## Code Quality and AI-Assisted Contributions

AI-assisted and agent-assisted contributions are welcome. However, contributors remain fully responsible for all submitted code.

Use Arena's [agent skills](https://github.com/isaac-sim/IsaacLab-Arena/tree/main/skills) for recurring contribution workflows. In particular, `/commit-and-pr` follows the project's commit and pull request conventions; see the [commit-and-pr skill](https://github.com/isaac-sim/IsaacLab-Arena/blob/main/skills/developer/commit-and-pr/SKILL.md). Review feedback from the Isaac Lab-Arena Review Bot as part of preparing your change, and address applicable findings. Automated review complements author self-review and maintainer review.

Before a pull request can be merged, it must:

- Be small (fewer than 500 changed lines). Breaking larger changes into a series of small pull requests increases the likelihood that maintainers will review them.
- Have gone through several rounds of author self-review for correctness, including generated code.
- Be focused on one coherent change.
- Follow project conventions and maintain API consistency.
- Include appropriate tests and documentation.
- Pass the `pre-commit` hooks and all required CI checks, and include benchmark results when the change affects performance.
- Address maintainer review feedback.

A pull request is not ready to merge solely because an agent generated it or because it works in one local environment. It must meet the same quality, reliability, and maintenance standards as every other contribution.

## Submitting a Pull Request

Read [CONTRIBUTING.md](https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTING.md) for the project's contribution rules, including coordinate-frame naming and commit sign-off instructions.

When opening a pull request:

- Link the related issue or Design Proposal.
- Explain what changed and why.
- Describe how the change was tested or validated.
- Keep the change focused; split unrelated work into separate pull requests.
- Break large features into smaller stacked pull requests, with each pull request describing its dependencies and contributing a coherent, reviewable change.
- Update documentation when user-facing behavior or public APIs change. For new features, include an example workflow and/or a concept documentation update when appropriate to the feature's scope and documentation needs.
- Sign off every commit with the Developer Certificate of Origin (DCO).

Thank you for contributing thoughtfully and for helping build IsaacLab-Arena with the community.
