"""Install Bash interactive bindings for virtual environments and Pixi shells."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


_HOOK_START = "# >>> seamless-mode-venv-hook >>>"
_HOOK_END = "# <<< seamless-mode-venv-hook <<<"
_HOOK = f"""{_HOOK_START}
case "$-" in
  *i*)
    if [ -n "${{BASH_VERSION:-}}" ] && command -v seamless-mode-bind.sh >/dev/null 2>&1; then
      . "$(command -v seamless-mode-bind.sh)"
    fi
    ;;
esac
{_HOOK_END}
"""


def _activation_script(prefix: Path | None) -> Path:
    if prefix is None:
        prefix_text = os.environ.get("VIRTUAL_ENV")
        if prefix_text:
            prefix = Path(prefix_text)
        elif os.environ.get("CONDA_PREFIX"):
            raise SystemExit(
                "This is a Conda environment; its activation hook is installed by seamless-suite."
            )
        else:
            prefix = Path(sys.prefix)

    prefix = prefix.expanduser().resolve()
    activation = prefix / "bin" / "activate"
    bind_script = prefix / "bin" / "seamless-mode-bind.sh"
    if not activation.is_file():
        raise SystemExit(f"No POSIX venv activation script found at {activation}")
    if not bind_script.is_file():
        raise SystemExit(
            f"{bind_script} is missing; install seamless-transformer in this environment first."
        )
    return activation


def _install_hook(activation: Path) -> None:
    content = activation.read_text(encoding="utf-8")
    has_start = _HOOK_START in content
    has_end = _HOOK_END in content
    if has_start and has_end:
        print(f"Seamless mode hook is already installed in {activation}")
        return
    if has_start or has_end:
        raise SystemExit(f"Incomplete Seamless mode hook markers in {activation}")

    if content and not content.endswith("\n"):
        content += "\n"
    content += "\n" + _HOOK
    activation.write_text(content, encoding="utf-8")
    print(f"Installed Seamless mode hook in {activation}")


def _remove_hook(activation: Path) -> None:
    content = activation.read_text(encoding="utf-8")
    start = content.find(_HOOK_START)
    end = content.find(_HOOK_END)
    if start == -1 and end == -1:
        print(f"No Seamless mode hook found in {activation}")
        return
    if start == -1 or end == -1 or end < start:
        raise SystemExit(f"Incomplete Seamless mode hook markers in {activation}")

    line_start = content.rfind("\n", 0, start) + 1
    block_end = end + len(_HOOK_END)
    if block_end < len(content) and content[block_end] == "\n":
        block_end += 1
    content = content[:line_start] + content[block_end:]
    activation.write_text(content, encoding="utf-8")
    print(f"Removed Seamless mode hook from {activation}")


def venv_hook_main() -> None:
    parser = argparse.ArgumentParser(
        description="Install or remove the interactive Bash hook in a POSIX venv."
    )
    parser.add_argument("action", choices=("install", "remove"))
    parser.add_argument(
        "--prefix",
        type=Path,
        help="venv path; defaults to VIRTUAL_ENV or the current Python prefix",
    )
    args = parser.parse_args()
    activation = _activation_script(args.prefix)
    if args.action == "install":
        _install_hook(activation)
    else:
        _remove_hook(activation)


def shell_main() -> int:
    """Launch an interactive Bash with the Seamless bindings loaded."""
    if os.name == "nt":
        raise SystemExit("seamless-mode-shell requires Bash on a POSIX platform")
    bash = shutil.which("bash")
    bind_script = shutil.which("seamless-mode-bind.sh")
    if not bash:
        raise SystemExit("bash was not found on PATH")
    if not bind_script:
        raise SystemExit("seamless-mode-bind.sh was not found; install seamless-transformer first")

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="seamless-mode-", suffix=".bashrc", delete=False
    ) as rc_file:
        rc_path = Path(rc_file.name)
        rc_file.write('if [ -r "$HOME/.bashrc" ]; then\n  . "$HOME/.bashrc"\nfi\n')
        rc_file.write("if ! declare -F seamless-mode-bind >/dev/null 2>&1; then\n")
        rc_file.write(f"  . {shlex.quote(bind_script)}\n")
        rc_file.write("fi\n")

    try:
        return subprocess.call([bash, "--rcfile", str(rc_path), "-i"])
    finally:
        rc_path.unlink(missing_ok=True)
