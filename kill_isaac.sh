#!/bin/bash

# Find and kill Isaac-related processes
echo "Searching for Isaac-related processes..."
pids=$(ps aux | grep -i 'isaac' | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
  echo "No Isaac processes found."
else
  echo "Killing the following Isaac processes:"
  echo "$pids"
  kill -9 $pids
  echo "Done."
fi
