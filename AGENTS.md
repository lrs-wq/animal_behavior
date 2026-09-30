# Eating Behavior Agent Instructions

## Scope gate

- Apply the knowledge workflow only when the request requires reading or changing this repository.
- For general conversation or work unrelated to this repository, do not run knowledge commands and do not update knowledge files.

## Knowledge bootstrap

- Start repository work with `python3 /root/project-knowledge/bin/kg context . --task "<current request>"`.
- Read `docs/knowledge/CURRENT.md` before changing inference, events, identity, configuration, or the stream viewer.
- Use graph `evidence` paths to load only the source relevant to the task.
- The live Git state is essential here: check uncommitted configuration, deployment, model, test, and viewer changes before editing overlapping files.

## Task routing

- Runtime entry and orchestration: `src/eating_behavior/main.py` and `src/eating_behavior/pipeline.py`.
- Backend user/RTMP registry and event delivery: `src/eating_behavior/alerts.py`, `docs/backend_event_delivery.md`, and `deploy/systemd/eating-behavior-events.service`.
- Customer/source identity mapping: `src/eating_behavior/config.py`, `src/eating_behavior/identity.py`, and `README.md`.
- Detection, tracking, pose and posture: `src/eating_behavior/models.py`, `tracking.py`, `pose.py`, and `posture.py`.
- Event state machines: `events.py`, `eating_state.py`, `presence_state.py`, and `posture.py`.
- Runtime configuration: `configs/pipeline.yaml` and `configs/models.yaml`.
- Viewer: `web/stream-viewer/README.md` and `web/stream-viewer/`.
- Offline training only: `training/README.md` and `training/`.

## Knowledge update contract

- Update `docs/knowledge/GRAPH.json` when a model, source mapping, module, event, service, config, deployment, or relationship changes.
- Update `docs/knowledge/CURRENT.md` when supported behavior, model promotion rules, current event semantics, or known gaps change.
- Append one semantic event to `docs/knowledge/CHANGES.jsonl`; do not record formatting-only edits.
- Add an ADR under `docs/knowledge/DECISIONS/` for a durable architecture or model-governance choice.
- Keep knowledge changes in the same commit as the code they describe.
- Before finishing any repository modification, run `python3 /root/project-knowledge/bin/kg diff .`, update every affected semantic area, then run `python3 /root/project-knowledge/bin/kg refresh .`, `python3 /root/project-knowledge/bin/kg check .`, and `python3 /root/project-knowledge/bin/kg gate .`.
- A `stale` or `unbaselined` gate result means the task is not complete. `knowledge_update_pending` is acceptable only before committing; include source and knowledge files in the same commit.
- After a commit that changes source or knowledge, run `python3 /root/project-knowledge/bin/kg gate . --require-current`.

## Server code and data layout

- Keep active project source, executable Python model configs, tests, deployment templates and knowledge under /root/Eating_behavior.
- Keep model weights, datasets, media, events, SQLite user/source registry and outbox, persistent runtime output, test outputs, caches and backups under /data/Eating_behavior. Production and manual server runs must set their output/cache directories there; do not rely on relative output defaults in the source checkout.
- Production model weight paths must be regular files under /data/Eating_behavior/models with the existing SHA-256 verification. Executable pose config remains under /root/Eating_behavior/models/rtmpose; it is code and is also checksum-pinned.
- Viewer must execute the /root/Eating_behavior/web/stream-viewer source through a read-only systemd bind at /opt/joyai-stream-viewer. The latter is a service-private mountpoint, not a separately maintained code copy. Preserve the dedicated non-root user and ProtectHome.
- /opt/eating-behavior/conda-env and /usr/bin/node are installed language runtimes/dependencies; /etc holds installed units and credentials. These are infrastructure, not project data directories.
- /root/Eating_behavior/data may be a convenience directory symlink to /data/Eating_behavior. Verify physical destinations when auditing; do not put persistent business data in /run.
- Before storage changes, preserve existing files, verify copied hashes and runtime access, update related readers and model allowed roots, and validate service startup. Do not replace model contents or relax artifact validation for a path migration.

## Safety and verification

- Run `python -m pytest -q` for Python changes.
- Run `npm --prefix web/stream-viewer run check` and `npm --prefix web/stream-viewer test` for viewer changes.
- Runtime inference must not silently promote training output; follow the controlled model path and checksum process.
- Never add stream credentials, customer mappings, raw frames/videos, model weights, W&B data, or production events to knowledge files.
