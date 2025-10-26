import json
from pathlib import Path

EMB_DIR = Path("face_db/embeddings")
manifest_file = Path("face_db/manifest.json")
manifest_file.parent.mkdir(parents=True, exist_ok=True)

# Load existing manifest
if manifest_file.exists():
    with open(manifest_file) as f:
        manifest = json.load(f)
else:
    manifest = {"students": []}

# Load student_ids.json
student_ids_file = EMB_DIR / "student_ids.json"
with open(student_ids_file) as f:
    student_ids = json.load(f)

# Add new students
for sid in student_ids:
    if sid not in manifest["students"]:
        manifest["students"].append(sid)

# Save manifest
with open(manifest_file, "w") as f:
    json.dump(manifest, f, indent=2)

print("[INFO] Manifest updated with all enrolled students!")
