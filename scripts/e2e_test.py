#!/usr/bin/env python3
"""End-to-end smoke test against a *live* running instance of the service --
not pytest, and not mocked. Exercises the real HTTP API exactly as a client
would: create -> chunked upload -> interrupt -> resume -> complete -> poll
status -> semantic search -> ownership isolation -> list -> delete.

Usage:
    python3 scripts/e2e_test.py              # starts docker compose if needed
    python3 scripts/e2e_test.py --no-compose # assumes the API is already running

Prints one short PASS/FAIL line per step, plus a final summary. Exits
non-zero if anything failed.

Uses httpx, already a project dependency (see requirements.txt) for the
TestClient in tests/ -- no extra install needed to run this.
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SAMPLE_FILE = SCRIPT_DIR / "sample.log"

API_URL = "http://localhost:8000/v1"
HEALTH_URL = "http://localhost:8000/health"
USER_A = "e2e-user-a"
USER_B = "e2e-user-b"
CHUNK_SIZE = 400  # deliberately small so the sample file needs several chunks

passed = 0
failed = 0


def ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"  [PASS] {msg}")


def bad(msg: str) -> None:
    global failed
    failed += 1
    print(f"  [FAIL] {msg}")


def step(title: str) -> None:
    print(f"\n== {title} ==")


def info(msg: str) -> None:
    print(f"  {msg}")


def die(msg: str) -> None:
    bad(msg)
    summary()
    sys.exit(1)


def summary() -> None:
    print(f"\n== Summary: {passed} passed, {failed} failed ==")


def start_compose() -> None:
    step("Starting docker compose")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        die("docker compose up failed")
    ok("stack started")


def wait_for_health() -> None:
    step("Waiting for the API to report healthy")
    for i in range(1, 61):
        try:
            resp = httpx.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                ok(f"health check ok after ~{i * 2}s")
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    die("service never became healthy within 120s")


def main() -> None:
    if "--no-compose" not in sys.argv:
        start_compose()

    if not SAMPLE_FILE.exists():
        die(f"sample file missing at {SAMPLE_FILE}")

    wait_for_health()

    data = SAMPLE_FILE.read_bytes()
    file_size = len(data)
    info(f"sample file: {SAMPLE_FILE} ({file_size} bytes)")

    client = httpx.Client(timeout=30)

    # ---- 1. create upload ---------------------------------------------------

    step("Register upload")
    resp = client.post(
        f"{API_URL}/files",
        headers={"X-User-Id": USER_A},
        json={"filename": "sample.log", "total_size": file_size},
    )
    if resp.status_code != 201:
        die(f"POST /files returned {resp.status_code}: {resp.text}")
    file_id = resp.json()["file_id"]
    ok(f"created upload, file_id={file_id}")

    # ---- 2. chunked upload with a simulated interrupt + resume -------------

    step("Chunked upload with a simulated interruption")
    offset = 0
    chunk_index = 0
    interrupted = False
    while offset < file_size:
        chunk_index += 1

        # Halfway through, stop without sending the rest -- simulates a
        # dropped connection -- then confirm /status reports the exact
        # resume point before continuing.
        if not interrupted and offset >= file_size // 2:
            interrupted = True
            status_resp = client.get(f"{API_URL}/files/{file_id}/status", headers={"X-User-Id": USER_A})
            reported_offset = status_resp.json()["bytes_received"]
            if reported_offset == offset:
                ok(f"interrupted after {offset} bytes; /status agrees on resume offset")
            else:
                bad(f"resume offset mismatch: sent {offset}, /status says {reported_offset}")

        end = min(offset + CHUNK_SIZE, file_size)
        piece = data[offset:end]

        resp = client.put(
            f"{API_URL}/files/{file_id}/chunk",
            params={"offset": offset},
            headers={"X-User-Id": USER_A, "Content-Type": "application/octet-stream"},
            content=piece,
        )
        if resp.status_code != 200:
            die(f"chunk {chunk_index} (offset={offset}) rejected with HTTP {resp.status_code}: {resp.text}")
        offset = end
    ok(f"uploaded {file_size} bytes across {chunk_index} chunks")

    # ---- 3. complete ---------------------------------------------------------

    step("Complete the upload")
    resp = client.post(f"{API_URL}/files/{file_id}/complete", headers={"X-User-Id": USER_A})
    upload_status = resp.json().get("upload_status")
    if upload_status == "completed":
        ok("upload_status=completed")
    else:
        bad(f"expected upload_status=completed, got '{upload_status}' (response: {resp.text})")

    # ---- 4. poll status until searchable -------------------------------------

    step("Poll status until indexing catches up")
    processing_status = None
    chunks_indexed = 0
    for i in range(1, 61):
        resp = client.get(f"{API_URL}/files/{file_id}/status", headers={"X-User-Id": USER_A})
        body = resp.json()
        processing_status = body.get("processing_status")
        chunks_indexed = body.get("chunks_indexed", 0)
        if processing_status == "completed":
            ok(f"processing_status=completed ({chunks_indexed} passages indexed, ~{i}s)")
            break
        time.sleep(1)
    else:
        bad(f"indexing did not finish within 60s (last processing_status={processing_status})")

    # ---- 5. the assignment's own semantic search example ---------------------

    step("Semantic search (the assignment's own example)")
    resp = client.post(
        f"{API_URL}/files/{file_id}/search",
        headers={"X-User-Id": USER_A},
        json={"query": "database connectivity problems", "top_k": 3},
    )
    results = resp.json().get("results", [])
    top_text = results[0]["text"] if results else ""
    if "Connection to database failed" in top_text:
        ok(f"query 'database connectivity problems' -> top hit is the DB error line (score={results[0]['score']})")
    else:
        bad(f"expected top search hit to mention the DB error, got: {top_text!r}")

    # ---- 6. ownership isolation ------------------------------------------------

    step("Ownership isolation")
    resp = client.get(f"{API_URL}/files/{file_id}/status", headers={"X-User-Id": USER_B})
    if resp.status_code == 404:
        ok("a different X-User-Id gets 404 for this file")
    else:
        bad(f"expected 404 for a non-owner, got HTTP {resp.status_code}")

    # ---- 7. list -----------------------------------------------------------------

    step("List uploads for the owner")
    resp = client.get(f"{API_URL}/files", headers={"X-User-Id": USER_A})
    found = any(f["file_id"] == file_id for f in resp.json().get("files", []))
    if found:
        ok("uploaded file appears in GET /files")
    else:
        bad("uploaded file missing from GET /files")

    # ---- 8. delete ----------------------------------------------------------------

    step("Delete the file")
    resp = client.delete(f"{API_URL}/files/{file_id}", headers={"X-User-Id": USER_A})
    if resp.status_code == 204:
        ok("DELETE returned 204")
    else:
        bad(f"expected 204 from DELETE, got HTTP {resp.status_code}")

    resp = client.get(f"{API_URL}/files/{file_id}/status", headers={"X-User-Id": USER_A})
    if resp.status_code == 404:
        ok("file is gone after delete (404 on status)")
    else:
        bad(f"expected 404 after delete, got HTTP {resp.status_code}")

    summary()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
