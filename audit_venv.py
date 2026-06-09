"""
Audit which packages take the most space in your venv.

Usage:
  python audit_venv.py
"""
import os
import sys
from importlib import metadata


def folder_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def main():
    results = []
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue

        # Find the actual folder(s) on disk for this package
        files = dist.files or []
        roots = set()
        for f in files:
            try:
                abs_path = str(dist.locate_file(f))
                # Take the top-level folder under site-packages
                if "site-packages" in abs_path:
                    rel = abs_path.split("site-packages" + os.sep, 1)[1]
                    top = rel.split(os.sep, 1)[0]
                    roots.add(os.path.join(
                        abs_path.split("site-packages")[0],
                        "site-packages",
                        top,
                    ))
            except Exception:
                pass

        size = 0.0
        for r in roots:
            if os.path.exists(r):
                size += folder_size_mb(r)

        if size > 0.1:
            results.append((size, name))

    results.sort(reverse=True)

    total = sum(s for s, _ in results)
    print(f"\n{'Size (MB)':>12}  Package")
    print("-" * 50)
    for size, name in results[:50]:
        print(f"{size:>12.1f}  {name}")

    print("-" * 50)
    print(f"{total:>12.1f}  TOTAL ({len(results)} packages)\n")


if __name__ == "__main__":
    main()
