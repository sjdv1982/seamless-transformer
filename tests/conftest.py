"""Shared runtime images for Phase 3 language-backend contracts."""

from pathlib import Path
import subprocess

import pytest


RUNTIME_ONLY_BINARY = "seamless-phase3-runtime-only"
R_IMAGE = "seamless-phase3-r:local"
JULIA_IMAGE = "seamless-phase3-julia:local"
R_BASE_IMAGE = "rocker/r-ver:4.4.2"
JULIA_BASE_IMAGE = "julia:1.12.6"


def _run(argv, *, timeout=900):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="session")
def language_runtime_images(tmp_path_factory):
    """Build Docker images once, then load/convert them for all three engines."""
    context_root = tmp_path_factory.mktemp("phase3-language-images")
    julia_context = context_root / "julia"
    julia_context.mkdir()
    repo_root = Path(__file__).resolve().parents[2]
    julia_project = repo_root / "seamless" / "julia"
    for filename in ("Project.toml", "Manifest.toml"):
        (julia_context / filename).write_bytes((julia_project / filename).read_bytes())
    (julia_context / "Dockerfile").write_text(
        "\n".join(
            (
                f"FROM {JULIA_BASE_IMAGE}",
                "ENV JULIA_DEPOT_PATH=/opt/phase3-julia-depot",
                "ENV JULIA_CPU_TARGET=generic",
                "WORKDIR /opt/phase3-julia-project",
                "COPY Project.toml Manifest.toml ./",
                "RUN julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'",
                f"RUN printf '#!/bin/sh\\necho {RUNTIME_ONLY_BINARY}\\n' > /usr/local/bin/{RUNTIME_ONLY_BINARY} \\",
                f"    && chmod 755 /usr/local/bin/{RUNTIME_ONLY_BINARY} \\",
                "    && chmod -R a+rX /opt/phase3-julia-depot",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    r_context = context_root / "r"
    r_context.mkdir()
    (r_context / "Dockerfile").write_text(
        "\n".join(
            (
                f"FROM {R_BASE_IMAGE}",
                "RUN apt-get update && apt-get install -y --no-install-recommends \\",
                "    build-essential zlib1g-dev \\",
                "    && rm -rf /var/lib/apt/lists/*",
                "RUN Rscript --vanilla -e \"install.packages(c('jsonlite', 'RcppCNPy'), \\",
                "    repos='https://cloud.r-project.org')\"",
                f"RUN printf '#!/bin/sh\\necho {RUNTIME_ONLY_BINARY}\\n' > /usr/local/bin/{RUNTIME_ONLY_BINARY} \\",
                f"    && chmod 755 /usr/local/bin/{RUNTIME_ONLY_BINARY}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    _run(["docker", "info"], timeout=180)
    _run(["podman", "info"], timeout=180)
    _run(["apptainer", "--version"], timeout=60)
    _run(["docker", "build", "-t", R_IMAGE, str(r_context)])
    _run(["docker", "build", "-t", JULIA_IMAGE, str(julia_context)])

    image_archive = context_root / "phase3-language-images.tar"
    _run(["docker", "save", "-o", str(image_archive), R_IMAGE, JULIA_IMAGE])
    _run(["podman", "load", "-i", str(image_archive)], timeout=300)

    sif_dir = context_root / "apptainer"
    sif_dir.mkdir()
    for image in (R_IMAGE, JULIA_IMAGE):
        sif = sif_dir / (image + ".sif")
        _run(
            ["apptainer", "build", "--fakeroot", str(sif), "docker-daemon:" + image],
            timeout=900,
        )
        assert sif.is_file()
    return {"r": R_IMAGE, "julia": JULIA_IMAGE, "sif_dir": sif_dir}
