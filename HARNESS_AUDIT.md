# mypibypy harness audit — 2026-10-26

> **Status 2026-10-27:** all eight steps of section G are applied. Steps 1–5:
> suspended servers killed, port pinned to 8533, `indexes.local.json` created, MCP
> server registered with Claude Code and Claude Desktop, `client.py`
> sampler/keep-alive/timeout message, `ToolError` refusals with a wire test, and
> the `app.py` changes (no-change guard, step timeline, disabled input mid-turn,
> model-health line, `width="stretch"`, persisted Form run record). Steps 6–8:
> `EVALUATE --unscreened--> FLAGGED` settles an unscreened design without a model
> call, and `FLAGGED --override--> SUCCEEDED` is the MCP client's explicit
> `proceed`; the Form's agent loop re-runs the real pipeline and offers only
> `reduce_pairs` / `switch_system` (`DesignAgent(allowed_actions=…)`; any other
> model choice stops for a human instead of looping); the intent schema is
> `change` / `explain` / `unclear` with no `relax` field; the README drift is
> fixed; and the battery is `scripts/benchmark_chat.py` over
> `agent/battery.py`, with tests. The full suite is 772 passed, 9 opt-in skips.

Scope: the local-model harness (agent loop, chat path, Ollama client), the Streamlit
GUI and the MCP server, as they run on this MacBook (18 GB M3 Pro, conda env
`pifish-tools`, Ollama 0.32.15 serving `qwen3.5:9b`). Everything below was measured
on this machine today; nothing in the repository was changed. The scripts used are in
this session's scratchpad (`chat_probe.py`, `sampler_ab.py`, `mcp_e2e.py`) and are
worth keeping as `scripts/benchmark_chat.py` (see F.1).

## 1. The short version

The harness itself works. The decision call, the chat intent call, the state machine,
the MCP tools and the test suite all behave as designed when they are given the model
on an otherwise idle machine. What makes it *look* broken is four things around it,
in this order of likelihood:

1. **You are probably looking at a dead server.** Seven `streamlit run app.py`
   processes exist on this machine. Six are suspended (ctrl-Z, state `T`) but still
   hold ports 8501–8506, so a tab on `localhost:8501` connects to a process that will
   never answer. The only live one (pid 36593, started 2026-10-20 15:23) is on
   **8507**. Section A.1.
2. **Memory.** The model is 5.7 GB wired; swap sat at 7.7–9.0 GB of 9–10 GB during
   the audit. Under that pressure one chat turn went from ~10 s to **two 120 s
   timeouts**, and the MCP server took 14 s to start instead of 0.9 s. The six
   suspended Streamlit processes are part of that load. Section A.2.
3. **Setup gaps, not code.** `indexes.local.json` does not exist, so the Form and the
   chat have no index to pick even though hg38 and GENCODE v50 bowtie indexes are on
   disk. The MCP server has never been registered with Claude Code or Claude Desktop
   (Claude Desktop's `mcp.log` is empty). Section A.3.
4. **Two real code defects.** The MCP server's carefully written refusals never reach
   the client: the SDK reduces every `ValueError`/`ParameterRefused` to
   `Error executing tool X`. And the chat accepts an "adjust" that changes nothing and
   silently designs the same thing again while the model's comment promises a change.
   Sections E.1 and C.2.

Then there is a list of smaller harness, GUI and MCP improvements (sections B–F).

## 2. What was verified working (evidence)

| check | result |
|---|---|
| Ollama model | `qwen3.5:9b` Q4_K_M, 5.7 GB, 100 % GPU, `RENDERER qwen3.5` so the system prompt is honoured (pirate test passed) and `think:false` works: 0.3 s vs 3.5 s with 180 chars of thinking |
| Decision call (`OllamaDecisionClient.decide`) | cold 12.6 s, warm 4.5 s on a 654-token prompt; chose `relax_tm` for a Tm-limited nothing-passes case, 3/3 usable |
| Chat intent call, tuned sampler, idle machine | 9/10 intents as expected, 6.8–16.3 s per turn (mean 9.9 s); every "explain" reply passed `reply_check`; "why is pair 2 worse than pair 1?" correctly answered that pair 2 is better |
| Form agent loop as `app.py` runs it (unscreened design) | `flagged` after 1 model call, 7.5 s, with and without `enforce_window` |
| Test suite with the env's `bin` on PATH | **729 passed, 9 skipped (opt-in live tests)** in 4 m 36 s. Without PATH, 49 skips and 1 false failure (`blastn not on PATH`) |
| Headless app boot (`AppTest`) | 0 exceptions; sidebar, chat, design panel render |
| Live app on 8533 | pasted 774 nt, designed 1 pair in the Form: map, table, close-up, download all rendered |
| MCP server over stdio | 7 tools advertised; `design_target` on a real hg38 locus, 3 pairs, screened with bowtie: **5.1 s, state `succeeded`**; `diagnose_design`, `try_change`, `write_order` fine; run record persisted |
| Network | UCSC and Ensembl reachable (no TLS proxy problem today) |

## A. Environment

### A.1 Seven Streamlit servers, six of them suspended

```
pid    port  state                       started
58022  8501  suspended (T)               Oct 19 12:45
7165   8502  suspended (T)               Oct 20 12:15
12915  8503  suspended (T)               Oct 20 12:59
18394  8504  suspended (T)               Oct 20 13:43
23454  8505  suspended (T)               Oct 20 14:11
24699  8506  suspended (T)               Oct 20 14:15
36593  8507  live, foreground, tty s006  Oct 20 15:23
78210  8533  this audit's preview        today
```

Streamlit takes the next free port when 8501 is busy, so each new `streamlit run`
landed one port higher while the browser tab stayed where it was. A suspended process
keeps its listening socket; connections queue and never complete. The suspended ones
also hold ~300–500 MB each in swap.

Fix: from the terminal that owns them run `jobs` and `kill %N`, or:

```bash
kill -9 58022 7165 12915 18394 23454 24699
```

Then run exactly one server on a fixed port. The repo's `.claude/launch.json` already
pins 8533; for a hand-started one use `streamlit run app.py --server.port 8533`.
Consider `server.port = 8533` in `mypibypy/.streamlit/config.toml` so a stray start
fails loudly instead of drifting to a new port.

### A.2 Memory is the real ceiling

| measurement | value |
|---|---|
| RAM | 18 GB |
| model resident | 5.7 GB wired (`ollama ps`) |
| swap during audit | 7.7 of 9.2 GB → 9.0 of 10.2 GB |
| MCP server start, idle | 0.9 s |
| MCP server start, while the test suite loaded the hg38 bowtie index | 14.2 s |
| chat turn, idle | 7–16 s |
| chat turn, same load | 7 s → 47 s → **120 s timeout ×2** |

The pipeline's own bowtie screen loads a ~3 GB hg38 index, so a screened design plus
the resident model plus several Streamlit processes plus Claude Desktop is already
past physical RAM. Things that help, cheapest first:

- Kill the suspended servers (A.1).
- Set `keep_alive` on each request (B.2): the model then stays resident for the
  working session instead of reloading after 5 idle minutes (8 s cold penalty), and
  unloads when you stop.
- Halve the KV cache: run `ollama serve` with `OLLAMA_FLASH_ATTENTION=1` and
  `OLLAMA_KV_CACHE_TYPE=q8_0`. At `num_ctx` 8192 the f16 cache is ~1.1 GB; q8 is
  ~0.55 GB.
- If it still swaps, try a smaller model with the battery script before changing code:
  the harness only needs enum choice and a seven-field intent. A 4B Qwen3.5 at Q4 is
  ~2.5 GB. `MYPIBYPY_LOCAL_MODEL` already switches it.

> **Measured 2026-10-30** with `scripts/benchmark_chat.py --loop`, one model resident
> at a time, pinned sampler unless said. `qwen3.5:4b` (Q4_K_M, 3.3 GB pulled, 3.4 GB
> resident against the 9B's 5.7 GB) loaded without growing swap, where loading the 9B
> grew it from 2.9 to 6.5 GB, and ran the turns ~35 % faster (7–8.6 s against 10.7 s
> warm). It is not a drop-in. Pinned sampler: 9/10 in 3 of 3 runs, always the same
> turn -- "why is pair 2 worse than pair 1?" is answered correctly and then loops
> ("please clarify which property ...") to the 512-token limit, so the user waits ~23 s
> for the cut-off error. Modelfile sampler (presence_penalty 1.5): 10/10 twice, clean
> reply checks, but it turns `enforce_window` on for "loosen the Tm", which the battery
> does not score on that turn. Pinned plus `repeat_penalty 1.1`: the loop goes, but
> "make it 6 pairs" comes back with `n_pairs` unchanged while the note promises six --
> the penalised token is the 6, as the SAMPLING comment predicts. `qwen3.5:2b` (Q8_0,
> 2.5 GB resident): 3× faster, 9/10, and it swaps pair 1's and pair 2's penalties to
> agree with the question (only its inverted accessibility comparison is caught by
> `reply_check`) and echoes the current pair count and spacing instead of "unchanged".
> The default stays `qwen3.5:9b` (10/10, 10.7 s/turn warm, as on 10-27); switching to
> the 4B is a sampler decision, not a model-name change.

### A.3 Setup gaps

**No `indexes.local.json`.** The sidebar says "pick an index in the Form", but the
Form's "Screen against" menu holds only *Default — off* and *Off*, and the chat's
`screen` enum is empty, so the instruction is a dead end. Both indexes exist on disk.
Create `mypibypy/indexes.local.json` (gitignored):

```json
{"indexes": {
  "hg38":        {"path": "/Users/sstaller/scripts_ss/genomes/hg38/hg38",
                  "kind": "genome", "description": "human GRCh38"},
  "gencode-v50": {"path": "/Users/sstaller/scripts_ss/genomes/gencode_v50/transcripts",
                  "kind": "transcriptome", "description": "GENCODE v50 transcripts"}
}}
```

After that "screen against hg38" works in the chat (verified: the model returns
`screen: "hg38"` when the name is offered), and RNA targets get the transcriptome.

**MCP server never registered.** `~/.claude.json` has no `mcpServers`, there is no
`.mcp.json`, and Claude Desktop's `mcp.log` is empty. The server works (section 2), it
just has never been launched by a client. Register it with the real paths:

```bash
claude mcp add mypibypy --scope user \
  -e PATH="$HOME/miniforge3/envs/pifish-tools/bin:/usr/bin:/bin" \
  -e MYPIBYPY_BOWTIE_INDEX="$HOME/scripts_ss/genomes/hg38/hg38" \
  -e MYPIBYPY_TRANSCRIPTOME_INDEX="$HOME/scripts_ss/genomes/gencode_v50/transcripts" \
  -- "$HOME/miniforge3/envs/pifish-tools/bin/python" -m mypibypy mcp
```

For Claude Desktop the same command/args/env go into
`~/Library/Application Support/Claude/claude_desktop_config.json` under
`mcpServers.mypibypy` (the README's block is right; fill in these paths). Note the
Desktop app here is an enterprise install; if the entry does not appear, org policy
on MCP servers is the next thing to check, not the server.

**PATH.** Anything that runs the package outside an activated env (an MCP client, a
launchd job, this audit's first test run) must prepend the env's `bin`, or bowtie,
blastn and RNAplfold are silently "not on PATH" and every pair is NOT SCREENED. The
MCP registration above does this; the README says so; it is worth a runtime check in
`status` (E.4).

## B. The harness (agent layer)

### B.1 Sampler options are inherited from the Modelfile

The Modelfile ships `temperature 1, top_k 20, top_p 0.95, presence_penalty 1.5`. The
client overrides only `temperature`. A presence penalty still moves logits under
greedy decoding and under grammar constraints, and it penalises tokens seen in the
last 64 tokens of context, which at the start of a reply is the end of the prompt:
exactly the numbers the `note` is told to repeat verbatim.

Measured (chat battery, same ten turns):

| sampler | intents as expected | mean s/turn | notes |
|---|---|---|---|
| stock options (as shipped today), under concurrent load | 6/10 | 35.7 | two 120 s timeouts |
| `presence_penalty 0, repeat_penalty 1.0, top_k 1`, idle | 9/10 | 9.9 | no timeouts |

The stock run was confounded by the memory load in A.2, so I re-ran the four most
sensitive turns alone, alternating the two settings, twice each:

| sampler, idle machine | mean s/turn | max | correctness |
|---|---|---|---|
| stock options | 8.6 | 15.3 | "why did I only get 4?" refused by `reply_check` on **both** repeats (it said windows were "rejected" while the window is not enforced), so the app would make a second model call each time; first request labelled `adjust` |
| tuned (`presence_penalty 0, repeat_penalty 1.0, top_k 1`) | 8.7 | 15.6 | all eight replies accepted first time |

So the 120 s timeouts were memory, not sampling: **speed is identical when the
machine is idle.** The sampler is a quality lever: with the stock options one of the
four turns reliably needs the retry round trip (another 6–13 s) or falls back to the
software's facts.

Recommendation: set the sampler explicitly in `request_body` so the harness does not
depend on whatever Modelfile Ollama pulled:

```python
SAMPLING = {"temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "presence_penalty": 0.0, "repeat_penalty": 1.0}
"options": {**SAMPLING, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
```

and pin it with a test next to `TestThinkIsMandatory`.

### B.2 No `keep_alive`

Ollama unloads after 5 idle minutes by default; the first call after that pays the
load (12.6 s vs 4.5 s measured). Add `"keep_alive": "30m"` to the request body (it is
per request, no server config needed). Thirty minutes keeps a working session warm
and frees 5.7 GB when you walk away.

### B.3 The Form's agent loop is a demonstration, not a loop

In `app.py:form_editor`, `design_fn` returns the same `(primary, candidates)`
whatever parameters the loop passes, and `Budget(max_turns=2)`. So:

- any non-terminal action (`relax_tm`, `reduce_pairs`, …) re-"designs" the identical
  result, re-evaluates to the same `NEEDS_JUDGEMENT`, asks the model again, and on the
  second apply hits the turn budget: the user sees **"Agent: failed — turn budget of 2
  exhausted" over a perfectly good design**. Today the model happened to pick
  `flag_for_review` for the unscreened case, which is why this is not seen every time;
  for the nothing-passes case it picked `relax_tm`, which would trigger it.
- for genomic targets the window stats are computed from a re-scan of the raw region
  without repeat masking, so the counts the model reads differ from the pipeline's
  ranking; for RNA and pasted targets only the top 5 candidates are passed, so the
  evidence reads "0/5 windows".

Fix: either make `design_fn` call the pipeline with the loop's parameters
(`design_for_target(target, p, backend)` already exists for the chat), or give the UI
loop only terminal actions (a `Budget(max_turns=1)` plus a client wrapper that maps
relaxations to `flag_for_review`), and say in the UI which one it is.

### B.4 "Unscreened" should not cost a model call

`classify` sends every unscreened design to the model. On this machine with no index
configured that is a 5–8 s call per Form design whose only defensible answer is
`flag_for_review` (and that is what the model returned, twice). A missing screen is a
missing input, not a conflict between signals. Add an edge
`EVALUATE --unscreened--> FLAGGED` to `states.TRANSITIONS` and return it from
`classify` deterministically; keep the model for genuine tradeoffs. This also makes
the MCP server's behaviour simpler (a design with no backend goes straight to
`flagged` with the reason, and `decide_design proceed` from `flagged` could be the
explicit override).

### B.5 Timeouts and failure messages

`OllamaDecisionClient(timeout=120)` returns "could not reach Ollama … timed out" on a
slow machine, which sends the user to `ollama serve` when the server is up and
swapping. Distinguish `TimeoutError`/`socket.timeout` from connection refused, and
say "the model took longer than 120 s; the machine is probably short of memory (see
`ollama ps`)". Consider a shorter timeout for the chat intent call (60 s) and the
`submit_mode="disable"` fix in D.3 so a second message cannot queue behind it.

### B.6 Seam inconsistencies

- `DecisionClient` (the Protocol) declares only `decide`, but the chat requires
  `structured`. `PydanticAIDecisionClient` has no `structured`, so
  `MYPIBYPY_DECISION_CLIENT=pydantic_ai` would crash the chat with `AttributeError`.
  Either add `structured` to the Protocol and the PydanticAI client, or have
  `build_client` refuse that kind for the chat.
- On the OpenAI-compatible `/v1` path `num_ctx` is not applied, so the server default
  (4096) governs; the chat prompt already overflowed 4096 once. Another reason to keep
  the chat on the native client.
- The run record is durable only for MCP. The Form's agent runs with `store=None`, and
  the chat never runs the loop, so `agent_runs.local.jsonl` does not exist on this
  machine despite the README's "every tool call, clamp and decision lands in a durable
  record". Pass `store=RunStore()` in `form_editor` (path resolved beside
  `pyproject.toml` the way `app.py` resolves the other local files).

## C. The chat path

### C.1 Intent accuracy on the real model

With the tuned sampler the model got channel, pair count, spacing, screen and
`enforce_window` right every time, answered questions from `facts`/`pair_comparison`
without inventing numbers, and pushed back when the question's premise was wrong. Two
quirks, both harmless today: it labels a fresh request `adjust` instead of `design`
about half the time (the app treats both the same, see C.3), and it sometimes fills
`system` with the *current* channel (equal to the current value, so no change).

### C.2 An "adjust" that changes nothing designs again anyway

Help and the README both offer *"too few passed, loosen the Tm"* as an example. Since
`CHAT_RELAXATIONS` was narrowed to `("none", "reduce_pairs")` the model has no field
for that, so it returns `intent: adjust` with every field at its sentinel and
`note: "I will loosen the Tm constraints and run the design again."` The app then
designs the same parameters again and shows run N+1, identical to run N, under a
model comment promising a change. The README's own sentence ("when a relaxation is
asked for while the window is not enforced, the reply says so") describes code that
no longer exists.

Fix in `chat_editor`: if `intent.wants_design`, `changes` is empty and a run on this
target exists, do not design; reply "Nothing in that request changes a setting the
design uses (pairs, spacing, channel, screen, GC/Tm enforcement), so the design was
not re-run" and list what *would* change it (`try_changes` already produces this).
Then drop the stale example from `app_help.py` and the README.

### C.3 The schema carries dead weight

`design` vs `adjust` is not acted on (`wants_design` covers both), and `relax` has a
single reachable value whose effect the model could express as `n_pairs`. Every extra
enum is another field a 9B model must fill correctly on every turn. Collapsing to
`intent ∈ {change, explain, unclear}` and dropping `relax` shrinks the schema and the
prompt that restates it. Keep the tests that prove no target/sequence field exists.

### C.4 Latency is legible nowhere

A chat turn is 7–16 s of model time plus 1–5 s of design plus, when `reply_check`
objects, a second model call; the user sees one spinner saying "Reading the request…"
for all of it. See D.2 for the Streamlit fix.

## D. The Streamlit GUI

The app follows the project's own rules well (no sequence on disk, NOT SCREENED in
words, shared session). Against the installed Streamlit 1.66 and its bundled skill
(`.agents/skills/developing-with-streamlit`) these stand out:

### D.1 Deprecated API in use

Every design render logs five `use_container_width` deprecation warnings (removal
date 2025-12-31 has passed). Replace with `width="stretch"` on the three
`st.altair_chart` calls and two `st.dataframe` calls in `render_result`. `st.radio(…,
horizontal=True)` for the target type wraps to a vertical stack in the narrow column
anyway; `st.segmented_control` is the recommended widget and fits.

### D.2 Make the long turn visible

Use `st.status(type="step")` containers inside the assistant message so the user sees
"Reading the request" → "Designing against …" → "Checking the answer" as they happen,
with the measured seconds in each label, instead of a single spinner. The chat-ui
reference shows the pattern. Add `submit_mode="disable"` to `st.chat_input` so a
second message cannot be sent during a 20 s turn (today it queues and both run).

### D.3 Model health in the sidebar

The sidebar says "Model: ollama" whether or not the server is up, the model is pulled,
or it is loaded. One cached call to `GET /api/tags` and `GET /api/ps` (ttl 30 s) gives
"qwen3.5:9b · loaded, 5.7 GB" or "Ollama not reachable at 127.0.0.1:11434" before the
first chat message fails. The same spot can show the free-memory warning from A.2.

### D.4 Smaller items

> **Done 2026-10-30.** `max_entries=64` on both sequence caches; Material icons
> replace the emoji in the sidebar, Help, run record, chat replies and the download
> button; the `offline` test fixture now seeds one generator (it produced a
> homopolymer before). The close-up is the exception: `st.image` caps an image at
> the column's width (`max-width: 100%`, even inside a horizontal container), so a
> 740 px base-by-base drawing rendered at 447 px; it stays an `<img>` data URL in a
> scrolling div, now through `st.html` rather than `unsafe_allow_html`.

- `cached_region` / `cached_transcript` have no `max_entries`; add one (sequence data
  stays in process memory for the server's lifetime otherwise).
- `show_svg` injects HTML with `unsafe_allow_html=True`; `st.image` accepts an SVG
  string directly (`st.image(svg, width="content")`), which keeps the native-size
  rendering without raw HTML.
- Charts and dataframes lack `alt=`; the skill asks for one on each.
- The status lines use emoji; Material icons (`:material/check_circle:`) are the
  recommended style. Cosmetic.
- The run picker, close-up and "All columns" expander are good; consider
  `on_change="rerun"` on the "All columns, folding and windows" expander so the full
  table is only built when opened.
- A "Screen against" entry built by scanning a `genomes/` directory for `*.1.ebwt`
  would have found the two indexes on this machine without any file.

## E. The MCP server

### E.1 Refusals are invisible to the client

The server raises `ValueError` and `ParameterRefused` with precise guidance ("call
`list_targets` for the ids", "only applies while a design is in 'decide'", "Designs do
not survive a server restart"). The MCP SDK 2.3 treats any exception other than
`ToolError` as a crash: the client receives `Error executing tool write_order` and the
reason is logged server-side only. Verified over stdio today. So the state-machine
gating works, but Claude cannot read why it was refused and cannot self-correct; it
will retry or give up. The tests never caught this because they assert refusals
in-process with `pytest.raises` and the one stdio test only checks the happy path.

Fix: wrap each tool body:

```python
from mcp.server.mcpserver.exceptions import ToolError

def _refusals_as_tool_errors(fn):
    @functools.wraps(fn)
    def inner(*a, **kw):
        try:
            return fn(*a, **kw)
        except (ValueError, ParameterRefused, IllegalTransition) as exc:
            raise ToolError(str(exc)) from exc
    return inner
```

and add a wire test: `call_tool("write_order", {"design_ids": ["d_nothing"]})` over
stdio must return `is_error=True` with "unknown design id" in the text.

### E.2 Registration and startup

Never registered (A.3). Startup is 0.9 s idle but 14 s under memory load; Claude
Desktop's launch timeout is generous, but `status` should report memory and PATH
problems rather than letting the first `design_target` return NOT SCREENED.

### E.3 Designs live only in memory

`write_order` only knows designs made since the server started, and clients restart
servers often. Persist `StoredDesign` (parameters, run id, state; the sequence can be
re-derived by re-running the deterministic pipeline, which is cheap) in a small
SQLite or JSONL beside the run record, keyed by `design_id`, so a workbook can be
written in a later conversation. Keep the sequence out of it: re-design on demand.

**Applied 2026-10-29:** `mcp_store.py` keeps `mcp_designs.local.jsonl` beside the run
record (`$MYPIBYPY_MCP_DESIGNS`), one JSON line per state change with the target id,
parameters, screen flag, pad assignment, run id, warnings, stop reason, machine position
and a SHA-256 digest of the oligos and verdicts -- never a sequence. A `Session` that
does not know an id rebuilds it from its entry by running the pipeline again, refuses it
if the digest differs (loci entry, reference or screen configuration changed), and
otherwise keeps the saved state, so a reloaded `flagged` design still needs `proceed`.
Tests: `tests/test_mcp_server.py::TestDesignsSurviveARestart`.

### E.4 Tool surface suggestions

- `status` should include `bowtie_on_path`, `rnaplfold_on_path`, free memory and the
  Ollama state; it is the first call every client makes.
- Expose the named indexes from `indexes.local.json` as a `screen` enum on
  `design_target` (the Form and chat already use this mechanism), instead of one
  genome and one transcriptome from env.
- Expose the run record and `diagnose_design` output as MCP *resources*
  (`run://<run_id>`), so a client can read a run's history without a tool call.
- Return `structured_content` schemas (`output_schema`) for `design_target` and
  `diagnose_design`; the SDK supports it and clients render typed results better.
- `decide_design` from `flagged` as the explicit human-override path once B.4 lands.
- A read-only `list_runs` over `agent_runs.local.jsonl` (ids, targets, statuses) for
  "what did we design last week".

**Applied 2026-10-29:** `status` reports which of bowtie, bowtie-build, blastn,
RNAplfold and RNAfold are on the server's PATH (`shutil.which`), the PATH itself, free
memory and swap (`vm_stat` and `sysctl` on macOS, `/proc/meminfo` on Linux, a sentence
elsewhere), and the named indexes from `indexes.local.json` with whether each can run;
`design_target` takes `screen_against`, a name from that file resolved server-side to a
bowtie backend for exactly that index (`specificity.bowtie_backend`), refused when
unknown (with the configured names), of the wrong kind for the target (the Form's
sentence, shared through `indexes.kind_mismatch`) or unable to run; and the run record
and a design's diagnosis are the resources `run://<run_id>` and
`design://<design_id>/diagnosis`, filtered through `DNA_RUN`. Not done: the Ollama
state (the MCP server does not use Ollama), `output_schema`, `list_runs`. `proceed`
from `flagged` was already in place from the state-machine gating. Tests:
`tests/test_mcp_server.py::TestStatusReportsTheMachine`, `::TestNamedScreens`,
`::TestResources`.

## F. General and documentation

### F.1 Keep the battery as a benchmark

`scripts/benchmark_chat.py` (from this audit's `chat_probe.py`): ten scripted chat
turns on a synthetic sequence, scored against expected intent fields, with timing,
`reply_check` outcomes and a sampler/model switch. It is the fastest way to answer
"is the harness working" and "can a 4B model do this" without touching lab data. The
`scripts/validate_agent.py` already does this for the decision path; the chat path has
no equivalent today.

### F.2 Documentation drift

- README and Help still advertise "loosen the Tm" from the chat (C.2) and the README's
  chat transcript shows `Applied: Tm window 55.0-62.0 → 53.5-63.5 C`, which the chat
  can no longer produce.
- README says "every tool call, clamp and decision lands in a durable record"; true
  for MCP only (B.6).
- README's registration block uses placeholder paths; A.3 has the real ones.
- The test count in the README (512) is stale; 738 collected.

### F.3 Things that are fine and should stay

The guarantees are structurally enforced and tested: no sequence field in any model
schema, closed enums, bounds in code, the state machine driving both the loop and the
MCP tools, `think: false` on the native endpoint, `reply_check` grounding every number.
None of the problems above touch those. The deterministic pipeline is fast (1–5 s) and
the model is used sparingly; with the sampler, keep-alive and memory fixes the chat is
a 10 s turn on this laptop, which is usable.

## G. Suggested order of work

1. Kill the suspended servers; run one on 8533. (A.1) — minutes
2. Create `indexes.local.json`; register the MCP server. (A.3) — minutes
3. `client.py`: explicit sampler, `keep_alive`, timeout message. (B.1, B.2, B.5)
4. `mcp_server.py`: `ToolError` wrapping plus a wire test. (E.1)
5. `app.py`: no-change guard, `st.status` steps, `submit_mode`, `width="stretch"`,
   persist the Form's run record, model-health line. (C.2, D.1–D.3, B.6)
6. `loop.py`/`states.py`: unscreened → flagged without a model call; real `design_fn`
   in the Form or terminal-only actions. (B.3, B.4)
7. Schema diet and docs drift. (C.3, F.2)
8. Benchmark script in `scripts/`. (F.1)
