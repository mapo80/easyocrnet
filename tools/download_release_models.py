"""Download EasyOCR.NET release models into an unpacked directory."""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Dict

import urllib.request


REPO = "mapo80/easyocrnet"
DEFAULT_TAG = "v2025.09.19"

ASSETS: Dict[str, str] = {
    "easyocrnet-models-cpu-onnx.zip": "onnx",
    "easyocrnet-models-openvino-ir.tar.gz": "openvino",
}


def _api_request(url: str, token: str | None) -> dict:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"token {token}")
    request.add_header("User-Agent", "easyocrnet-packager")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _download_asset(asset: dict, destination: Path, token: str | None) -> Path:
    url = asset["browser_download_url"]
    name = asset["name"]
    target = destination / name
    if target.exists() and target.stat().st_size == asset["size"]:
        print(f"✓ {name} already downloaded")
        return target

    print(f"↓ Downloading {name} ({asset['size'] / (1024 * 1024):.1f} MB)")
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"token {token}")
    request.add_header("User-Agent", "easyocrnet-packager")
    with urllib.request.urlopen(request) as response, target.open("wb") as fh:
        fh.write(response.read())
    return target


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(destination)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination)


def download_models(tag: str, output: Path, token: str | None) -> None:
    release = _api_request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", token
    )

    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    missing = sorted(name for name in ASSETS if name not in assets)
    if missing:
        raise RuntimeError(f"Missing assets in release {tag}: {', '.join(missing)}")

    output.mkdir(parents=True, exist_ok=True)

    for name, subdir in ASSETS.items():
        asset = assets[name]
        archive = _download_asset(asset, output, token)
        target_dir = output / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"↺ Extracting {name} into {target_dir}")
        if name.endswith(".zip"):
            _extract_zip(archive, target_dir)
        elif name.endswith(".tar.gz"):
            _extract_tar(archive, target_dir)
        else:
            raise ValueError(f"Unsupported archive format for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Git tag to download")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("external") / "release-models",
        help="Destination directory for extracted models",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token (defaults to GITHUB_TOKEN env var)",
    )
    args = parser.parse_args()

    download_models(args.tag, args.output, args.token)


if __name__ == "__main__":  # pragma: no cover
    main()
