# Prism emulator (on demand)

Boots an Android emulator on GitHub Actions, installs the published Prism app
(com.nahid002345.prismunitconverter) and runs commands that Claude sends through
the `prism-emulator-bridge` branch: launch, tap, type, screenshot, UI dump,
screen recording, stop.

Start a session by pushing a new `control/start.json`; the session stops itself
after `idle_minutes` without commands or `max_minutes` in total.
Client: `tools/emu.py`.
