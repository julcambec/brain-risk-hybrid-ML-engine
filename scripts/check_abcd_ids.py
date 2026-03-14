import pathlib
import re
import sys

pattern = re.compile(rb"NDAR_INV[A-Za-z0-9]{8}")
matches: list[str] = []

for path in sys.argv[1:]:
    p = pathlib.Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        continue

    try:
        data = p.read_bytes()
    except Exception:
        continue

    for match in pattern.finditer(data):
        matches.append(f"{path}:{match.group().decode('ascii')}")

if matches:
    sys.stderr.write("ABCD subject IDs found!\n")
    sys.stderr.write("\n".join(matches) + "\n")
    raise SystemExit(1)
