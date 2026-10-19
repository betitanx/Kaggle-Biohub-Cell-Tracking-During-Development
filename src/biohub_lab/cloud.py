"""Pure configuration helpers for remote Modal execution.

This module intentionally does not import ``modal``.  The control plane can
validate and render a remote job locally, while the optional Modal app owns
the cloud SDK integration.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Optional


_TRANSIENT_REMOTE_ERROR_RE = re.compile(
    r"(?:429|too many requests|rate limit|5\d\d)", re.IGNORECASE
)


def normalise_gpu_request(value: Optional[str]) -> Optional[str]:
    """Return a Modal GPU type, or ``None`` for an explicit CPU worker."""

    raw = str(value or "").strip()
    if raw.lower() in {"", "cpu", "none"}:
        return None
    return raw


def is_transient_remote_error(detail: str) -> bool:
    """Return whether an external API error is safe to retry."""

    return bool(_TRANSIENT_REMOTE_ERROR_RE.search(str(detail)))


def retry_delay_seconds(attempt: int) -> float:
    """Return bounded exponential backoff for a zero-based retry attempt."""

    return min(60.0, 2.0 ** max(0, int(attempt)))


def require_kaggle_api_token(environment: Mapping[str, str]) -> None:
    """Fail early when the remote Kaggle Secret has no usable token.

    The value is intentionally never returned, logged, or copied into the
    control-plane configuration.  This check only distinguishes a missing or
    blank Modal Secret from an authentication failure returned by Kaggle.
    """

    if not str(environment.get("KAGGLE_API_TOKEN", "")).strip():
        raise RuntimeError(
            "Modal Secret must define a non-empty KAGGLE_API_TOKEN; "
            "refresh the kaggle-api Secret before running remote staging"
        )


@dataclass(frozen=True)
class RemoteLayout:
    """Persistent paths inside the mounted remote Volume."""

    root: Path = Path("/mnt/biohub")

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train_split"

    @property
    def holdout_dir(self) -> Path:
        return self.data_dir / "holdout"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def state_path(self) -> Path:
        return self.root / "state" / "submissions.json"

    @property
    def lab_path(self) -> Path:
        return self.root / "lab.md"


@dataclass(frozen=True)
class RemoteRunConfig:
    """Safe, serialisable settings for one cloud cycle."""

    volume_name: str = "biohub-cell-tracking-data"
    gpu: str = "A100"
    data_scope: str = "full"
    duration_hours: float = 4.0
    live_submit: bool = False
    competition: str = "biohub-cell-tracking-during-development"
    remote_project_root: Path = Path("/root/biohub")
    volume_root: Path = Path("/mnt/biohub")
    config_path: Path = Path("config/experiments.json")

    def validate(self) -> None:
        if not self.volume_name.strip():
            raise ValueError("volume_name must not be empty")
        if not self.gpu.strip():
            raise ValueError("gpu must not be empty")
        if self.data_scope not in {"full", "subset"}:
            raise ValueError("data_scope must be 'full' or 'subset'")
        if self.duration_hours <= 0.0:
            raise ValueError("duration_hours must be positive")
        if self.live_submit and self.data_scope != "full":
            raise ValueError("live submission requires data_scope='full'")
        if not self.competition.strip():
            raise ValueError("competition must not be empty")
        if not str(self.config_path).strip():
            raise ValueError("config_path must not be empty")

    @property
    def layout(self) -> RemoteLayout:
        return RemoteLayout(Path(self.volume_root))

    def environment(self) -> dict[str, str]:
        """Return non-secret environment variables for the remote worker."""

        self.validate()
        layout = self.layout
        return {
            "KAGGLE_CONFIG_DIR": "/tmp/kaggle",
            "TEST_DATA_DIR": str(layout.test_dir),
            "DATA_DIR": str(layout.train_dir),
            "HOLDOUT_DATA_DIR": str(layout.holdout_dir),
            "BIOHUB_SUPPORT_PACK_DIR": str(layout.root / "support-pack"),
            "CELL_TRACKING_ROOT": str(self.remote_project_root / "external/kaggle-cell-tracking-competition"),
            "CELL_TRACKING_PYTHON": "/usr/local/bin/python",
            "BIOHUB_DATA_SCOPE": self.data_scope,
            "BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1" if self.live_submit else "0",
            "KAGGLE_DAILY_SUBMISSION_LIMIT": "5",
        }

    def command(self, output_dir: Optional[Path] = None) -> list[str]:
        """Build the CLI command executed inside the remote worker."""

        self.validate()
        layout = self.layout
        runs_dir = Path(output_dir) if output_dir is not None else layout.runs_dir
        command = [
            "/usr/local/bin/python",
            "-m",
            "biohub_lab.cli",
            "run",
            "--config",
            str(
                self.config_path
                if self.config_path.is_absolute()
                else self.remote_project_root / self.config_path
            ),
            "--duration-hours",
            str(self.duration_hours),
            "--output-dir",
            str(runs_dir),
            "--state",
            str(layout.state_path),
        ]
        if self.live_submit:
            command.append("--live-submit")
        return command


def remote_config_from_env(env: Optional[Mapping[str, str]] = None) -> RemoteRunConfig:
    """Read cloud settings without reading any secret value."""

    source = dict(env or {})
    duration = float(source.get("BIOHUB_REMOTE_DURATION_HOURS", "4"))
    config = RemoteRunConfig(
        volume_name=source.get("BIOHUB_MODAL_VOLUME", "biohub-cell-tracking-data"),
        gpu=source.get("BIOHUB_MODAL_GPU", "A100"),
        data_scope=source.get("BIOHUB_DATA_SCOPE", "full"),
        duration_hours=duration,
        live_submit=source.get("BIOHUB_REMOTE_LIVE_SUBMIT", "0") == "1",
        competition=source.get("KAGGLE_COMPETITION", "biohub-cell-tracking-during-development"),
        remote_project_root=Path(source.get("BIOHUB_REMOTE_PROJECT_ROOT", "/root/biohub")),
        volume_root=Path(source.get("BIOHUB_MODAL_VOLUME_ROOT", "/mnt/biohub")),
    )
    config.validate()
    return config
