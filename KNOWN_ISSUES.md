# Known Issues

This document records unresolved product and engineering problems. Each topic is an investigation
record, not an implementation specification.

## TUI resize and scrollback

Do not claim this issue is fixed unless a solution passes the real-tmux acceptance criteria below.

### Current decision

Wizolt keeps its ordinary input, activity, Ask, approval, and command-selector UI on the terminal's
primary screen. Completed output remains available in native terminal and tmux scrollback.

This preserves the product's preferred interaction model, but dynamic primary-screen regions can
leave blank rows or stale fragments after tmux reflows the pane. The artifacts are currently
accepted because the attempted fixes either deleted real history, remained unstable across
repeated resizes, or made selectors materially worse to use.

Exclusive viewers such as `/diff` may still use the alternate screen. Ordinary command selectors
must not be moved there again without satisfying this document's acceptance criteria.

### Symptoms still reproducible

#### Dynamic activity and thinking output

During a long-running or streaming turn, repeatedly zooming and unzooming a tmux pane can leave:

- old thinking or activity frames in the visible pane;
- duplicated fragments in scrollback;
- blank rows that accumulate after each resize; and
- stale copies of the prompt or divider.

The current resize re-anchor keeps the live prompt from steadily climbing toward the top of the
pane, but it does not make the underlying primary-screen text flow reversible.

#### Inline command selectors

Selectors such as `/provider`, `/model`, and `/effort` temporarily make the non-full-screen
prompt-toolkit application taller. After the selector closes, a later tmux zoom/unzoom can expose
the former region as a large blank area in both the pane and scrollback.

A typical reproduction is:

1. Start `wizolt --yolo` inside tmux with existing shell output above it.
2. Open `/effort`, choose an item, and return to the normal prompt.
3. Zoom and unzoom the pane several times. One successful cycle is not sufficient.
4. Inspect both the visible pane and tmux copy-mode history.

The selector remains inline today because the alternatives tested below were less safe or less
usable.

### Related issue that is only partially solved

Commit `79fbfc2` added the current resize re-anchor. It fixes one specific failure: after tmux moves
the already-rendered application during reflow, the prompt no longer trusts the drifted cursor
position and climbs upward on every cycle. `tests/test_tui_resize.py` models and protects that
invariant.

That fix does not remove physical rows already inserted into the primary-screen text flow. It also
does not prove that real tmux scrollback is unchanged. Treat prompt anchoring and scrollback
preservation as separate properties.

### Why this is difficult

Wizolt deliberately combines two different models:

- completed transcript is ordinary terminal output and belongs in native scrollback; and
- drafts, live activity, selectors, and status are a repaintable prompt-toolkit application at the
  bottom of the primary screen.

tmux reflows physical terminal rows when pane width changes. A logical application row can become
several physical rows, including rows carrying historical wrap state. prompt-toolkit knows its last
logical screen, but it cannot read the terminal screen or scrollback and therefore cannot determine
which reflowed rows still belong to the live application.

The available terminal operations do not close that information gap:

- Erasing clears cells but does not remove rows from the terminal's text flow. Redrawing a shorter
  region can therefore leave permanent empty rows.
- Deleting lines can shorten the flow, but deleting one row too many destroys the transcript
  immediately above the application. The exact excess row count drifts across repeated reflows.
- A cursor-position report provides the cursor row, not row ownership or the contents and wrap
  state of the rows above it.
- Width estimates derived from the previous render become inaccurate after tmux repeatedly splits
  and rejoins wrapped history.
- Resize delivery is not a single clean event: SIGWINCH and prompt-toolkit's own size check may
  both invoke resize handling around different rendered frames.
- Streaming thinking text changes while resize and repaint are in progress, making the old and new
  regions different even without reflow.

There is consequently no safe primary-screen command that means "remove exactly the old live UI
and preserve every transcript row".

### Attempts and outcomes

#### Bottom re-anchor and erase

Commits `79fbfc2` and `80fa665` explored re-anchoring, erasing, and repainting from the pane bottom.
This can keep the visible prompt and live region single-copy in slow or static cases. Under repeated
fast resizing it still accumulates blank rows because erase does not shorten the text flow.

The broader live-region change from `80fa665` was reverted by `cb88770`. Commit `8ec86d4` preserves
the detailed handoff for that unresolved investigation.

#### CSI line deletion

Deleting reflowed rows with CSI `M` produced clean results for an isolated resize. It failed over
long sequences because the estimated number of excess physical rows drifted. Under-deletion left
fragments; over-deletion consumed real transcript lines. This is not an acceptable trade-off.

#### Alternate screen for command selectors

Commit `7ea8536` moved ordinary command selectors to the alternate screen. It appeared correct on
the first selector and resize cycle, but later cycles could restore an incomplete primary screen
and remove existing shell or transcript history from view. An additional attempt to avoid erasing
the primary buffer before the switch did not stabilize repeated resize behavior.

The change was fully reverted by `14c25ea`.

#### Fixed-height compact selector

An uncommitted experiment replaced the normal input region with a three-row selector. It avoided
height growth by construction, but the selector was too small to compare options or read useful
previews. The implementation was discarded.

#### Floating selector

A second uncommitted experiment used a roughly ten-row prompt-toolkit `Float`, whose preferred
height does not contribute to the main layout. In practice the available region was still too
narrow in important terminal states, and real tmux testing still showed primary-history loss. The
unit model's stable preferred height was not evidence that tmux's physical history was preserved.
The implementation was discarded.

#### Full-screen application

Running the entire interactive application on the alternate screen would avoid most primary-screen
reflow artifacts and is the conventional design for full-screen TUIs. It would also remove the live
conversation from native terminal scrollback and make Wizolt behave more like an editor or system
monitor. That is a product-level trade-off, not a local resize fix, and has not been accepted.

### Testing limitations

`tests/test_tui_resize.py` uses a deterministic terminal model and is valuable for cursor anchoring,
but it cannot reproduce all of tmux's behavior:

- alternate-screen buffer restoration;
- persistent wrap flags in scrollback;
- tmux's exact reflow across multiple width transitions;
- ordering between tmux resize, SIGWINCH, CPR, and prompt-toolkit redraws; and
- the interaction between continuous streamed updates and reflow.

A single successful resize, an unchanged prompt-toolkit preferred height, or a clean visible pane
is not sufficient evidence. Scrollback must also be inspected after repeated cycles.

### Acceptance criteria for any future fix

A future implementation must use a real tmux integration test in addition to unit tests. At a
minimum it must demonstrate all of the following:

1. Seed the primary screen with uniquely numbered shell-history and transcript marker lines.
2. Exercise idle input, a long command selector, streaming thinking/activity, Ask, and approval.
3. Repeat at least 30 wide/narrow and tall/short resize cycles; include zoom/unzoom.
4. Run with both slow pauses and rapid resize while streamed content changes.
5. Preserve every marker exactly once in `tmux capture-pane -p -S -` output.
6. Keep the prompt, status, and active frame single-copy in the visible pane.
7. Show no unbounded growth in blank rows or total captured rows.
8. Remain correct after opening and closing the same selector more than once.
9. Keep selectors usable at small, normal, and large pane sizes, including previews and long option
   lists.
10. Preserve Ctrl-C, Esc, Enter, search, and resize behavior without hangs.
11. Work when tmux alternate-screen support is enabled and when it is disabled.
12. Preserve native transcript scrollback after the application exits.

The test must fail against the current implementation for the specific artifact it claims to fix.
Do not replace the real-tmux test with `DummyOutput`, preferred-height assertions, or a synthetic
terminal alone.

### Relevant code and tests

- `wizolt/tui/app.py`: modal layout, primary/alternate-screen transitions, animation, and resize
  re-anchoring.
- `wizolt/tui/views.py`: choice, Ask, activity, and divider fragments.
- `wizolt/cli/modals.py`: command-selector, Ask, approval, and exclusive-viewer entry points.
- `tests/test_tui_resize.py`: the modeled prompt re-anchor contract.
- `tests/test_tui_scrollback.py`: prompt-toolkit suspension and ordered scrollback output.
- `tests/test_diff_command.py`: alternate-screen capability behavior for the exclusive diff viewer.

### Reopening the problem

Do not reopen this as another small erase, CPR, cursor-move, or alternate-screen patch. A credible
next attempt needs at least one of these foundations:

- a real tmux harness that measures physical history throughout development;
- a prompt-toolkit change exposing enough physical-row ownership to delete safely;
- a terminal protocol or application architecture that no longer mixes mutable live rows with
  append-only transcript rows; or
- an explicit product decision to use a full-screen alternate-buffer UI and provide conversation
  history inside Wizolt instead of relying on native scrollback.

Until then, preserving transcript data takes priority over hiding resize artifacts.
