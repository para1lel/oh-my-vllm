# Plan: rebuild the DeepSeek V4.1 numerical fixtures without the external reference tree

Status: plan only. This document does not record implemented behavior.
Written 2026-11-03, after the maintainer deleted the external reference tree
recorded in the ignored `LOCAL.md` file.

## Why this plan exists

The frozen reference tree occupied 492 GiB.
Converted checkpoint shards took 487 GiB: four MP4 model shards and a tokenizer.
Capture directories took 3.1 GiB: 413 directories, one per operator family.
Other captures took 1.9 GiB: per-family order, mean, and endpoint captures.

The checkpoint shards are 487 of the 492 GiB.
They are not fixtures. They are the model converted to the MP4 layout.

`development/model/deepseek_candidate.py` reads 56 MiB of that tree.
That part is one directory of seven files, and one checkpoint digest file.
The 656 frozen checks compare endpoint vectors from that directory with a new
four-rank candidate run.

The `--model` option already takes the original checkpoint root.
No check reads a converted copy, except a run that compares shard bytes.

## Target design

1. **Small versioned fixtures.** Keep only the files that the checks read.
   Pack them into a bundle of at most a few hundred MiB.
   Add a manifest that lists the SHA256 of each file.

2. **No checkpoint copy.** The candidate run takes the model root and a
   manifest of expected shard digests.
   A conversion step verifies the original shards before it reads them.
   No converted copy stays after the run.

3. **A generator for each captured family.** Each fixture directory records
   the exact command, the seed, and the input case set.
   The bundle is then rebuildable from the checkpoint.
   Each generator is a repository script with a fixed seed.

4. **The gate keeps its meaning.** The 656 checks, the 0.01 tolerance, and the
   frozen manifests do not change.
   Only the storage location and the rebuild path change.


## Work items

1. Add `development/model/fixture_bundle.py`.
   It provides `pack`, `verify`, and `unpack` subcommands.
   `pack` writes the bundle and the digest manifest.
   `verify` checks a bundle before a gate run.
   `unpack` restores a bundle to a temporary directory.

2. Move the small fixture files into the bundle.
   Keep the directory names, so the candidate runner keeps its paths.

3. Replace the converted checkpoint with digest checks on the original shards.
   Record the check result in the run output.

4. Add a generator for each captured family that currently runs on a GPU.
   Record the seed, the case list, the Torch version, and the CUDA version.

5. Document the bundle location, the rebuild steps, and the expected sizes in
   `docs/models/deepseek_v41/evaluation.md` and its Chinese translation.
   Keep raw captures in an external directory.

6. Rebuild the bundle, then run the full 656 checks.
   Compare the result with the delivered reports.
   The new bundle replaces recorded evidence only after that comparison.


## Sizes after the change

| Item | Size |
|---|---:|
| Fixture bundle | about 0.4 GiB, expected |
| Original checkpoint | already present, not copied |
| Run output | external directory, deleted after the run |

## Risk

- A rebuilt bundle can differ from the deleted one by one bit, and that
  difference changes the gate. Compare the rebuild with the delivered report
  values, not the bundle digest alone.

- Some captures came from a release build of the reference driver. This
  repository cannot regenerate them, so those families keep their recorded
  endpoint vectors.

- The bundle must stay small enough for one workstation. Do not add capture
  directories that the checks do not read.
