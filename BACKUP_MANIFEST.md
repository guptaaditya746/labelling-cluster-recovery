# Backup manifest

This is a sanitized source snapshot of the labelling project. The previous Git history is omitted because environment example files contained populated credential and local-path fields; those fields have been blanked or made relative in this snapshot.

Excluded local artifacts:

- `logs/`: Slurm and application logs present in the source directory.
- `.venv/`, Python caches, and generated output/cache directories covered by `.gitignore`.

The original directory and its Git history were left unchanged. Copy any needed logs or generated state to separate storage before the cluster account is removed.
