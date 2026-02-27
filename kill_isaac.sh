#!/bin/bash

#!/bin/bash

# Keyword to search for (default: "Direct")
keyword=${1:-Direct}

echo "Searching for processes matching keyword: '$keyword'..."

# Find matching PIDs
pids=$(ps aux | grep -i "$keyword" | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
  echo "No matching processes found."
else
  echo "Killing the following processes:"
  echo "$pids"
  kill -9 $pids
  echo "Done."
fi

