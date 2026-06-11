#!/usr/bin/env python3
"""
git_push.py — One-command git add + commit + push.

Usage:
    python git_push.py "your commit message"
    python git_push.py                          # opens prompt for message

Behavior:
    1. git add .                  (stages everything)
    2. git commit -m "message"    (skips if nothing to commit)
    3. git push origin main       (pushes to GitHub)

Notes:
    - Run this from your repo root.
    - If you have nothing to commit, it still tries to push (in case you
      have unpushed commits from before).
"""

import subprocess
import sys
import os


def run(cmd, check=True):
    """Run a shell command, stream output live, return exit code."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def main():
    # ----- Get commit message -----
    if len(sys.argv) >= 2:
        # Joined with spaces in case user didn't quote: python git_push.py fix the bug
        msg = " ".join(sys.argv[1:])
    else:
        msg = input("Commit message: ").strip()
        if not msg:
            print("❌ Empty commit message — aborted.")
            sys.exit(1)

    # ----- Sanity check: are we in a git repo? -----
    if not os.path.isdir(".git"):
        print("❌ Not a git repo (no .git folder here). cd into your repo first.")
        sys.exit(1)

    # ----- 1. git add . -----
    run(["git", "add", "."])

    # ----- 2. git commit (allow empty so push still works if nothing changed) -----
    # First check if there's anything staged
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if status.stdout.strip():
        run(["git", "commit", "-m", msg])
    else:
        print("\nℹ️  Nothing to commit — pushing existing commits if any.")

    # ----- 3. git push -----
    run(["git", "push", "upstream", "main"])

    print("\n✅ Pushed to upstream/main successfully.")


if __name__ == "__main__":
    main()
