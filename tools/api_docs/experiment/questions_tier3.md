# Tier 3: extra difficult questions

Posed verbatim to fresh agents; ground truth is in `answers_tier3.md`.

## H01 (data-flow)

A task returns a success-rate metric from its metrics list. Follow that metric from the task to the number printed at the end of an evaluation: how is it turned into recorder and metric configuration at env-build time, where does the recorded data land on disk, and how is the final value computed from it?

## H02 (trace-sequence)

During an episode-limited rollout, one of several parallel envs terminates. In what order does Arena's env subclass do its per-episode bookkeeping when that env is reset, and under what condition does it skip the actual Isaac Lab reset of that env?

## H04 (precedence)

An Experiment Run can be rebuilt several times. How is the episode budget split across rebuilds when several limits could apply (which source wins), and what else changes per rebuild: seed, output file names?

## H05 (trace-sequence)

After the experiment runner has executed all Runs of a typed Experiment, what does it do next, in order, and what happens to the KPI export when one Run failed but the runner was told to keep going?

## H06 (extension-checklist)

I want an Experiment to emit a new kind of InfluxDB KPI records file. What do I have to write and which places must be touched so the runner picks it up, and what constraints do the records and the selection have to satisfy? Use an existing exporter as the template.

## H07 (error-path)

I construct a Run config that replays two recorded-variation JSONL files (one per rebuild), set the number of rebuilds to 3, and give it a step-based rollout limit of 500 steps. Which validation fails first and with what message? If I then drop the rebuild count, what fails next? And if I instead used an episode limit of 1, what would fail?

## H09 (data-flow)

A task declares two completion criteria sets in its termination config. At run time, how does that become the boolean 'success' termination signal each step: which term function is installed, what advances the progress state and how often, and how are the two criteria sets combined?

## H10 (trace-sequence)

Relation-based placement can reject layouts the robot cannot reach. How does that IK check get plugged into placement, when is it silently left out, which objects does it test, and how is it ordered relative to the cheap geometry checks?

## H11 (extension-checklist)

I want each line of the per-episode results JSONL to carry an extra field computed from the env at episode end. Where do I hook that in, what must my hook return, which names and keys are off-limits, and what has to be set for anything to be written at all?

## H12 (precedence)

An environment declares a default physics backend, the run passes a backend preset, and the environment also has a callback that edits the generated env config. Which backend is used, where is that decided, what does choosing Newton change beyond the physics config, and what stops the callback from switching backends?

## H13 (precedence)

The language instruction that reaches a policy can come from the run/CLI, from the task, or (for GR00T) from the policy's own YAML config. What is the precedence, how does the value reach the policy, and what happens when it ends up empty for GR00T versus the openpi-style remote policies?

## H14 (counting)

How many concrete variation classes does Arena ship (outside tests), how many are applied at run time through event terms versus once at build time, and which of them get attached to assets automatically without the environment author adding them?

## H15 (precedence)

In a typed YAML Experiment file, a value such as a rollout limit or the device can be given under the shared defaults, inside a Run, and as command-line overrides. In which order are these layers applied, which field does not merge, which override syntax is restricted, and what overrides the device in the end?

## H16 (data-flow)

If a DROID environment is built with the builder option that sizes cameras for the policy, and the policy is the openpi remote policy, what resolution do the DROID cameras end up rendering at, how is it computed, and what preconditions are checked? What would it be for the Cosmos policy?

## H17 (data-flow)

For the G1 whole-body-controller action term that takes direct upper-body joint targets, how is the incoming action vector laid out, how does each slice flow through the controller to joint targets, and which controller versions are supported?

## H18 (class-differences)

The VLM agent package offers two execution strategies on top of a shared VLM policy base: one that executes a list of commands per model call and one that drives a motion generator toward a predicted target. What are the concrete differences in config, when they ask the model again, how the next command is produced, and what extra context they send?

## H19 (error-path)

My environment graph YAML includes a second YAML file via its include key. Both files list an object with id 'mug', and the root file also has a relation whose subject is 'bowl', which is not defined anywhere. Which check fires first and with what message? What would fire instead if both 'mug' entries were in the same file?

## H20 (trace-sequence)

Walk through the single-environment policy runner entry point from parsing --policy_type to writing the report: how is the policy class found, where do its config flags come from, how is the rollout length decided, and what cleanup happens at the end?

## H21 (extension-checklist)

I am adding a new task type (e.g. 'press a lever'). Which methods must my class implement, which are optional with defaults, what must the termination method return, how do I make it discoverable, and what breaks if I report a success-rate metric without success criteria? Verify against an existing simple task.

## H22 (counting)

When placement runs with the cuRobo extension loaded, how many build-time placement checks are registered, in what order do they run by default, and which named placement check has no build-time validator at all? What names would I pass to restrict them?
