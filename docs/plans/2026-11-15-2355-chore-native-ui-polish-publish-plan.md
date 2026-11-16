---
title: Native UI Polish and Publish Readiness - Plan
type: chore
date: 2026-11-15
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
---

# Native UI Polish and Publish Readiness - Plan

## Goal Capsule

- **Objective:** A reader opening this repository sees the stage UI as it actually is today — default Scratch surface, ask bar, say bubble, and three variable monitors moved slightly right — with docs that describe the current product truthfully and a screenshot that shows the real stage.
- **Means:** Move the three monitor `x` values, fix the two doc files that still describe the removed "Bonsai progress" sprite, and capture a stage screenshot by the same headless-VM route already used this session. (KTD1, KTD2)
- **Authority:** The request is the authority: "Move the indicators to the right a bit and also tidy up the readmes and the documentation make this publish ready and screenshots and suhc". Nothing outside what that sentence names is in scope.
- **Execution profile:** one PR on `feat/bonsaiscratch-m0-m3`, then a fast-forward into `main` — the same pattern this branch has used.
- **Stop conditions:** If `tools/generate_project.py` has changed since the monitors were placed such that the build is not reproducible, stop and re-pin before touching docs. If no screenshot method succeeds after one full retry, ship the doc changes and record the screenshot gap in the PR body — do not block publish on it.
- **Who finishes and ships:** `ce-work` implements; `lfg` steps 3–10 carry the run to an open PR.

## Product Contract

### Summary

Make the publish-facing state truthful and tidy: shift the three analytics monitors a little right, correct the docs that still describe the old progress-panel UI, add a real screenshot to the README, and do no other work.

### Problem Frame

The UI changed three times this session and the docs were only partially updated. `README.md` and `docs/extension.md` still describe a "Bonsai progress" sprite with thirty costumes that no longer exists — the current `.sb3` has a single `Bonsai` sprite and three variable monitors (`layer`, `tokens`, `context`). A reader following the README would look for a sprite that is not there. The monitors also sit at stage x=240 (center); the request moves them a bit right.

### Requirements

- R1. The three monitors (`layer`, `tokens`, `context`) sit further right than x=240 in the generated project, at a single coordinate set chosen once and applied uniformly.
- R2. `README.md` and `docs/extension.md` contain no reference to a "Bonsai progress" sprite, thirty costumes, or a drawn progress panel; they describe the current ask-bar / say-bubble / monitors interface.
- R3. A real screenshot of the current stage appears in `README.md`, committed under `docs/` (e.g. `docs/stage.png`), showing the monitors, the ask bar, and a say bubble with text.
- R4. Both `.sb3` artifacts are rebuilt, `fixtures/artifact-checksums.json` is re-pinned, and `python3 -m pytest -q tests/test_artifacts.py` passes on the new builds.

### Scope Boundaries

**Outside this product's identity:** no redesign of the UI; the default-Scratch surface stands. No new variables, monitors, or costumes. No M5 decoder work, no performance re-measurement. No README rewrite beyond what correctness and the screenshot require.

**Deferred to Follow-Up Work:** a README section on how the monitors map to decoder state (nice, not requested); a `docs/screenshots/` convention if more images accumulate.

### Open Questions

- Whether the publish-facing README needs a one-line "the project is a research artifact, not interactive" disclaimer is left to the implementer's judgment — the request's "publish ready" is read as accurate-and-tidy, not as a marketing landing page.

## Planning Contract

### Key Technical Decisions

- KTD1. **Monitors move to x=320, y unchanged.** One coordinate set for all three (`layer`, `tokens`, `context`), applied in the single `for name, x, y in (...)` tuple in `tools/generate_project.py`. 320 keeps the default-width monitor (~100px) clear of the stage's right edge (480px wide, center 240) while visibly moving right. Rejected: x=380 — monitors' right edge would sit at the stage boundary and read as cramped; the request is "a bit", not "to the edge".
- KTD2. **Screenshot captured by the same headless-VM route used this session.** Open the editor, add `extension/loader.js` unsandboxed, load `dist/bonsai-prompt-to-text.sb3` by passing bytes to `vm.loadProject`, set `BS text`/`layer`/`tokens`/`context`, **click the green flag and wait for the `sensing_askandwait` prompt to appear**, then screenshot — the ask bar only renders inside the flag-started flow, not from the `BS show` broadcast. Rejected: a rendered mock presented as a screenshot — the request says screenshots and a fabricated capture is dishonest; also rejected: firing `BS show` alone, which renders the bubble but leaves the ask bar un-triggered (lite-review finding). If the route fails after one retry, the gap is recorded and the PR ships without it rather than blocking.
- KTD3. **Doc fixes are surgical, not a rewrite.** Only the passages that name the removed sprite/panel change; measured numbers, file paths, and benchmark tables stay verbatim. Rejected: a README restructure — out of scope and the doc is already accurate elsewhere.

### Assumptions

- The user is in a session with a working headless browser (prior turns did load the project), so a real screenshot is achievable this run; if not, the stop-condition in the Goal Capsule governs.
- "Publish ready" does not mean changing the project's research-artifact framing or adding a release/tag — just truthful docs and working artifacts.
- The `240, 8` / `240, 36` / `240, 64` tuples in `tools/generate_project.py` are the only place monitor positions are defined.
- `docs/` is the right home for a committed screenshot (consistent with `docs/plans/`, `docs/research/`).

### Sequencing

U1 (monitors) → U2 (rebuild + checksums) → U3 (doc fixes) → U4 (screenshot). U4 depends on U2 because the screenshot must show the moved monitors. U3 depends on U4 only for the image link target (`docs/stage.png` must exist for the README link to resolve); it is placed third so the screenshot lands in the README while the doc edits are open.

## Implementation Units

### U1. Move the three monitors right

**Goal:** The `layer`, `tokens`, and `context` monitors render a bit further right on the stage.

**Requirements:** R1.

**Files:** `tools/generate_project.py` (the `for name, x, y in (...)` loop that emits the `monitors` array).

**Approach:** Change each tuple's `x` from 240 to 320, keeping `y` and all other fields. One edit, three values.

**Test scenarios:**
- The generated `project.json` `monitors` array has `x: 320` for all three entries (verify by regenerating and reading the JSON).
- Monitor y-values and `visible: True` are unchanged.

**Verification:** Regenerate `dist/bonsai-prompt-to-text.sb3` and confirm the new `x` in the `monitors` block.

### U2. Rebuild artifacts and re-pin checksums

**Goal:** Both `.sb3` files reflect the moved monitors and their recorded checksums are accurate.

**Requirements:** R4.

**Dependencies:** U1.


**Approach:** Rebuild both artifacts (`python3 tools/generate_project.py dist/fp16-q8-scalar.sb3` and `... --text --benchmark`), rewrite `fixtures/artifact-checksums.json` with fresh SHA-256s, run `python3 -m pytest -q tests/test_artifacts.py`.

**Test scenarios:**
- `tests/test_artifacts.py` passes with zero failures on the new builds.
- The recorded hashes match the on-disk files byte-for-byte.

**Verification:** The test's own hash comparison is the proof; the unit is done when it is green.

### U3. Correct the docs that describe the old progress panel

**Goal:** `README.md` and `docs/extension.md` describe the current interface — ask bar, say bubble, three monitors — with no trace of the removed "Bonsai progress" sprite.

**Requirements:** R2.

**Files:** `README.md`, `docs/extension.md`.

**Approach:** In `README.md` (the "5. Run" paragraph and the "Bonsai progress sprite" paragraph) rewrite the status-panel text to name the monitors and the say bubble. In `docs/extension.md` remove or rewrite the "Bonsai progress sprite, whose thirty costumes..." claim to match the current `project.json` contents. Keep measured numbers, file paths, and tables verbatim. Add a `![Stage screenshot](docs/stage.png)` (or equivalent) reference where the README describes the running stage, pointing at the file U4 produces.

**Test scenarios:**
- `grep -n "Bonsai progress\|thirty costumes\|progress bar" README.md docs/extension.md` returns no matches for a live UI claim.
- The README names `layer`, `tokens`, `context` as the monitors.
- The README contains an image link whose target file exists after U4.

**Verification:** The greps come back clean for the old sprite and the image link resolves.

### U4. Capture and commit a real stage screenshot

**Goal:** A PNG of the actual stage — monitors moved right, ask bar, a say bubble with text — is committed and referenced.

**Requirements:** R3.

**Dependencies:** U2 (the screenshot must show the moved monitors, which requires the rebuilt project).

**Files:** `docs/stage.png` (new), `README.md` (the image link from U3).

**Approach:** Per KTD2 — headless browser, load the extension, load the project via `vm.loadProject(bytes)`, set display variables, click the green flag and wait for the ask prompt, screenshot the stage area. Crop to the stage if the frame includes the editor chrome. If the browser route fails after one full retry, record the gap and ship U1–U3 without the image, noting it in the PR body.

**Test scenarios:**
- `docs/stage.png` exists and is a valid PNG (> few KB, not a blank image).
- The image visibly contains the three monitors on the right, a say bubble with text, and the ask bar.

**Verification:** The PNG is committed and renders in the README diff view; visually confirms the moved monitors.

## Verification Contract

| Command | Proves |
|---|---|
| `python3 -m pytest -q tests/test_artifacts.py` | Both `.sb3` match their recorded checksums |
| `node --test tests/transport/*.test.js` | Transport/loader still works (already green; must stay so) |
| `grep -n "Bonsai progress\|thirty costumes" README.md docs/extension.md` | Stale UI claims are gone |
| `python3 -c "import zipfile,json; print([m['x'] for m in json.loads(zipfile.ZipFile('dist/bonsai-prompt-to-text.sb3').read('project.json'))['monitors']])"` | monitors `x` is 320 in the built project |
| `file docs/stage.png` | The screenshot is a real PNG committed to the repo |

## Definition of Done

- `tools/generate_project.py` emits monitors at x=320; `dist/bonsai-prompt-to-text.sb3` is rebuilt and its `project.json` carries the new values.
- `README.md` and `docs/extension.md` describe the current ask-bar / say-bubble / monitors interface with no "Bonsai progress" claim.
- `docs/stage.png` exists (or the gap is recorded per the stop condition) and is linked from the README.
- `tests/test_artifacts.py` is green; transport tests still pass.
- One PR on `feat/bonsaiscratch-m0-m3`, merged-or-pushed per the branch pattern, then fast-forwarded into `main`.
