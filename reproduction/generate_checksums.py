#!/usr/bin/env python3
"""Regenerate SHA-256 checksums for immutable release artifacts."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
paths=[]
for name in ('data','results','paper'):
    paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and not p.name.startswith('.'))
lines=[]
for p in sorted(paths):
    lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}")
(ROOT/'reproduction/checksums.sha256').write_text('\n'.join(lines)+'\n')
print(f"Wrote {len(lines)} checksums")
