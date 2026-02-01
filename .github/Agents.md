## Shell scritps
1. Always use GIT_REPO_ROOT=$(git rev-parse --show-toplevel) for relative paths.
2. No changes inside PatchTST_supervised folder.
3. Do not put import statement inside try catch. If cannot be imported, that shall error. 