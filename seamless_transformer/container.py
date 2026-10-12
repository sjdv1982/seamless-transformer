"""Container command construction for bash transformations."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
from copy import deepcopy
from typing import Any, Mapping


_CONTAINER_RUNTIME_NAMES = ("docker", "podman", "apptainer")
_CONTAINER_SPEC_KEYS = {"name", "gpus", "shm_size"}
_CONTAINER_RUNTIME_INFO_CACHE: dict[tuple[str, str | None, str | None, str | None], dict] = {}


def validate_container_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy the closed Docker image specification."""
    if not isinstance(spec, Mapping):
        raise TypeError("docker specification must be a mapping")

    if "options" in spec:
        raise ValueError(
            "Docker specification key 'options' is not supported; use 'gpus' "
            "and 'shm_size' instead"
        )
    if "version" in spec or "checksum" in spec:
        key = "version" if "version" in spec else "checksum"
        raise ValueError(
            "Docker specification key {!r} is not supported; use "
            "'name@sha256:<digest>' instead".format(key)
        )

    unknown = sorted(set(spec) - _CONTAINER_SPEC_KEYS)
    if unknown:
        raise ValueError("Unknown Docker specification key {!r}".format(unknown[0]))
    if "name" not in spec:
        raise ValueError("docker specification must contain 'name'")

    name = spec["name"]
    if not isinstance(name, str) or not name.strip():
        raise TypeError("docker specification 'name' must be a non-empty string")

    if "gpus" in spec and not isinstance(spec["gpus"], bool):
        raise TypeError("docker specification 'gpus' must be a bool")
    if "shm_size" in spec:
        shm_size = spec["shm_size"]
        if not isinstance(shm_size, str) or not shm_size:
            raise TypeError("docker specification 'shm_size' must be a non-empty string")

    return deepcopy(dict(spec))


def _remote_container_endpoint_error() -> str | None:
    for name in ("DOCKER_HOST", "CONTAINER_HOST"):
        value = os.environ.get(name)
        if value and not value.startswith("unix://"):
            return (
                "Remote container daemons are not supported; {}={!r} must name "
                "a local unix:// socket".format(name, value)
            )
    return None


def _runtime_info_cache_key(runtime: str) -> tuple[str, str | None, str | None, str | None]:
    return (
        runtime,
        os.environ.get("PATH"),
        os.environ.get("DOCKER_HOST"),
        os.environ.get("CONTAINER_HOST"),
    )


def _probe_container_runtime(runtime: str) -> dict[str, Any]:
    cache_key = _runtime_info_cache_key(runtime)
    cached = _CONTAINER_RUNTIME_INFO_CACHE.get(cache_key)
    if cached is not None:
        return cached

    executable = shutil.which(runtime)
    if executable is None:
        result = {"available": False, "rootless": None, "detail": "not found on PATH"}
        _CONTAINER_RUNTIME_INFO_CACHE[cache_key] = result
        return result

    if runtime == "docker":
        command = [executable, "info", "--format", "{{json .SecurityOptions}}"]
    elif runtime == "podman":
        command = [executable, "info", "--format", "{{.Host.Security.Rootless}}"]
    elif runtime == "apptainer":
        command = [executable, "--version"]
    else:
        raise ValueError("Unknown container runtime {!r}".format(runtime))

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {"available": False, "rootless": None, "detail": str(exc)}
    else:
        detail = (completed.stderr or completed.stdout).strip()
        if completed.returncode:
            result = {
                "available": False,
                "rootless": None,
                "detail": detail or "{} exited with status {}".format(
                    runtime, completed.returncode
                ),
            }
        else:
            if runtime == "docker":
                output = completed.stdout.strip()
                try:
                    security_options = json.loads(output)
                except (TypeError, ValueError):
                    security_options = output
                if isinstance(security_options, (list, tuple)):
                    rootless = any("rootless" in str(item).lower() for item in security_options)
                else:
                    rootless = "rootless" in str(security_options).lower()
            elif runtime == "podman":
                rootless = completed.stdout.strip().lower() == "true"
            else:
                rootless = False
            result = {"available": True, "rootless": rootless, "detail": detail}

    _CONTAINER_RUNTIME_INFO_CACHE[cache_key] = result
    return result


def container_runtime_info(runtime: str) -> dict[str, Any]:
    """Return cached availability and rootless status for a local runtime."""
    if runtime not in _CONTAINER_RUNTIME_NAMES:
        raise ValueError("Unknown container runtime {!r}".format(runtime))
    remote_error = _remote_container_endpoint_error()
    if remote_error is not None:
        raise RuntimeError(remote_error)
    info = _probe_container_runtime(runtime)
    if not info["available"]:
        raise RuntimeError(
            "Container runtime {!r} is unavailable: {}".format(runtime, info["detail"])
        )
    return dict(info)


def select_container_runtime(image_name: str | None = None) -> str:
    """Choose the first working local runtime, or honor the host setting."""
    configured = os.environ.get("SEAMLESS_CONTAINER_RUNTIME")
    hostname = socket.gethostname()
    description = (
        " for image {!r}".format(image_name) if image_name is not None else ""
    )

    if configured == "none":
        raise RuntimeError(
            "Container execution is disabled on host {!r} by "
            "SEAMLESS_CONTAINER_RUNTIME=none{}".format(hostname, description)
        )
    if configured is not None and configured not in _CONTAINER_RUNTIME_NAMES:
        raise ValueError(
            "Invalid SEAMLESS_CONTAINER_RUNTIME {!r}; expected docker, podman, "
            "apptainer, or none".format(configured)
        )

    remote_error = _remote_container_endpoint_error()
    if remote_error is not None:
        raise RuntimeError(remote_error)

    candidates = (configured,) if configured is not None else _CONTAINER_RUNTIME_NAMES
    failures = []
    for runtime in candidates:
        info = _probe_container_runtime(runtime)
        if info["available"]:
            return runtime
        failures.append("{}: {}".format(runtime, info["detail"]))

    if configured is not None:
        detail = failures[0] if failures else configured
        raise RuntimeError(
            "Container runtime {!r} is not available on host {!r}{} ({})".format(
                configured, hostname, description, detail
            )
        )
    raise RuntimeError(
        "No working container runtime is available on host {!r}{}; tried {}".format(
            hostname, description, "; ".join(failures)
        )
    )


def build_container_command(
    runtime: str,
    spec: Mapping[str, Any],
    job_directory: str,
    read_directories=(),
    *,
    tf_checksum: str | None = None,
    process_id: int | None = None,
    container_name: str | None = None,
    portable_job_directory: bool = False,
) -> list[str]:
    """Build a Docker or Podman argv for a materialized bash job."""
    spec = validate_container_spec(spec)
    if runtime not in ("docker", "podman"):
        raise ValueError("Runtime {!r} does not use the Docker/Podman command".format(runtime))

    pull_policy = os.environ.get("SEAMLESS_CONTAINER_PULL", "never")
    if pull_policy not in ("never", "missing"):
        raise ValueError(
            "Invalid SEAMLESS_CONTAINER_PULL {!r}; expected never or missing".format(
                pull_policy
            )
        )

    job_directory = os.path.abspath(os.path.expanduser(os.fspath(job_directory)))
    directories = []
    for directory in read_directories or ():
        directory = os.path.abspath(os.path.expanduser(os.fspath(directory)))
        if directory not in directories:
            directories.append(directory)
    for path in [job_directory, *directories]:
        if "," in path:
            raise ValueError(
                "Container bind mount paths cannot contain commas: {!r}".format(path)
            )
    mount_job_directory = (
        "__SEAMLESS_JOB_DIR__" if portable_job_directory else job_directory
    )

    command = [
        runtime,
        "run",
        "--rm",
        "--init",
        "--entrypoint",
        "",
        "--pull={}".format(pull_policy),
    ]
    if container_name is not None:
        command.extend(["--name", container_name])
    if process_id is not None:
        command.extend(["--label", "seamless.pid={}".format(process_id)])
    if tf_checksum is not None:
        command.extend(["--label", "seamless.tf={}".format(tf_checksum)])

    runtime_info = container_runtime_info(runtime)
    if runtime == "podman" and runtime_info["rootless"]:
        command.append("--userns=keep-id")
    elif not runtime_info["rootless"]:
        command.extend(["--user", "{}:{}".format(os.getuid(), os.getgid())])

    command.extend(
        [
            "--mount",
            "type=bind,src={},dst=/seamless-job".format(mount_job_directory),
            "-w",
            "/seamless-job",
            "--env",
            "HOME=/seamless-job",
        ]
    )
    for directory in directories:
        command.extend(
            [
                "--mount",
                "type=bind,src={0},dst={0},readonly".format(directory),
            ]
        )

    if spec.get("gpus", False):
        if runtime == "docker":
            command.extend(["--gpus", "all"])
        else:
            command.extend(["--device", "nvidia.com/gpu=all"])
    if "shm_size" in spec:
        command.extend(["--shm-size", spec["shm_size"]])
    try:
        command.extend(shlex.split(os.environ.get("SEAMLESS_CONTAINER_ARGS", "")))
    except ValueError as exc:
        raise ValueError("Invalid SEAMLESS_CONTAINER_ARGS: {}".format(exc)) from exc
    command.extend([spec["name"], "bash", "transform-inner.sh"])
    return command


def remove_container(runtime: str, container_name: str) -> None:
    """Force-remove a named Docker or Podman container, ignoring cleanup errors."""
    if runtime == "docker":
        command = ["docker", "rm", "-f", container_name]
    elif runtime == "podman":
        command = ["podman", "rm", "-f", "-t", "0", container_name]
    else:
        return
    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


__all__ = [
    "build_container_command",
    "container_runtime_info",
    "remove_container",
    "select_container_runtime",
    "validate_container_spec",
]
