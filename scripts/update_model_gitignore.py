#!/usr/bin/env python3
import pathlib, sys

# ----------------------------------------------------------------------
# Paths – adjust only if you move the repository
# ----------------------------------------------------------------------
repo_root = pathlib.Path(__file__).resolve().parents[1]   # two levels up from scripts/
model_dir = repo_root / "input_files/model_atmospheres/1D/marcs_standard_comp/marcs_standard_comp"
model_list_file = model_dir / "model_list"
gitignore_file = repo_root / ".gitignore"

# ----------------------------------------------------------------------
# Read model_list (ignore empty lines / comments)
# ----------------------------------------------------------------------
listed = []
with model_list_file.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        listed.append(line)

# ----------------------------------------------------------------------
# Build the .gitignore block
# ----------------------------------------------------------------------
ignore_block = [
    "\n# -------------------------------------------------",
    "# Keep ONLY the models listed in model_list",
    "# -------------------------------------------------",
    f"{model_dir.relative_to(repo_root) }/*"
]

for fn in listed:
    # Escape any leading ‘!’ or ‘#’ that might be part of the filename
    escaped = fn.replace('!', '\\!').replace('#', '\\#')
    ignore_block.append(f"!{model_dir.relative_to(repo_root)}/{escaped}")

# ----------------------------------------------------------------------
# Append (or replace) the block in .gitignore
# ----------------------------------------------------------------------
gitignore_text = gitignore_file.read_text()
# Simple approach: remove any previous block that starts with the marker line
start_marker = "# Keep ONLY the models listed in model_list"
if start_marker in gitignore_text:
    before, _sep, _after = gitignore_text.partition(start_marker)
    # keep everything before the old block, discard the old block, then add new one
    new_gitignore = before.rstrip() + "\n" + "\n".join(ignore_block) + "\n"
else:
    # just append at the end
    new_gitignore = gitignore_text.rstrip() + "\n" + "\n".join(ignore_block) + "\n"

gitignore_file.write_text(new_gitignore)
print(f"Updated .gitignore with {len(listed)} model entries.")