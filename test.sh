
#!/bin/bash
# run_tests.sh

# Ensure script exits if any command fails
set -e

# Run pytest with PYTHONPATH set to current directory
PYTHONPATH="$(pwd)" pytest --isolate -rx -v
