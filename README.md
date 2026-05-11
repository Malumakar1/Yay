# Circle Key Assist

Small Windows/Python helper for a local single-player timing prompt:

- captures a selected screen region
- detects the moving red segment and the static blue target segment
- reads the center digit with lightweight template matching
- presses that digit once when the red segment enters the blue target
- toggles on/off with `F8`

This is intended for your own local game/testing setup.

## Install

I already created `.venv` and installed the dependencies in this workspace. If you need to recreate it later:

```powershell
& 'C:\Users\harax\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On another Windows computer, you can also just double-click:

- `setup.bat` once, to create `.venv` and install dependencies
- `run.bat` whenever you want to run the app
- `run_dry_test.bat` to test detection without pressing keys
- `run_with_delay.bat` to run with an 8-24 ms random delay

## Change Hotkeys

Open `config.ini` in Notepad and change:

```ini
toggle_key = f6
quit_key = f12
```

Examples that work well: `f6`, `f7`, `f9`, `home`, `end`, `insert`, `delete`.

## Run

```powershell
.\.venv\Scripts\python.exe .\circle_key_assist.py --debug
```

On startup, drag a box around the whole circle prompt and press Enter.

Controls:

- `F8`: toggle active/inactive
- `F12`: quit
- `Esc` in the debug window: quit

## Check It Safely

Run a generated detection test:

```powershell
.\.venv\Scripts\python.exe .\circle_key_assist.py --self-test
```

Run against the real game without pressing keys:

```powershell
.\.venv\Scripts\python.exe .\circle_key_assist.py --debug --dry-run
```

Turn it on with `F8`. When the timing matches, the console should print `would press 3` or the current detected digit.

## Optional Tiny Delay

By default, the app presses as soon as the red segment enters the blue segment.

To add a short random delay, pass a millisecond range:

```powershell
.\.venv\Scripts\python.exe .\circle_key_assist.py --debug --delay-ms 8-24
```

To turn the delay off again, omit the flag or use:

```powershell
.\.venv\Scripts\python.exe .\circle_key_assist.py --debug --delay-ms off
```

If the red segment leaves the blue segment before the delay finishes, the queued press is cancelled.

## Useful Tuning

If your blue target is more cyan, try:

```powershell
.\.venv\Scripts\python.exe .\circle_key_assist.py --debug --blue-hue-min 80 --blue-hue-max 110
```

If your target is deeper blue, try:

```powershell
.\.venv\Scripts\python.exe .\circle_key_assist.py --debug --blue-hue-min 100 --blue-hue-max 135
```

If it presses too early or too late, adjust the target margin:

```powershell
.\.venv\Scripts\python.exe .\circle_key_assist.py --debug --inside-margin-deg -4
```

Negative margin waits until the moving segment is deeper inside the target.
