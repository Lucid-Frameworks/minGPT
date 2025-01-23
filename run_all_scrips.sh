#!/bin/bash

# Define script names
scripts=("generate_en.py" "generate_fr.py" "generate_it.py" "generate_es.py")
log_file="output_log.txt"

# Clear the log file if it exists
> $log_file

# Run each script and append its output to the log file
for script in "${scripts[@]}"
do
  echo "Running $script..." | tee -a $log_file
  python3 $script >> $log_file 2>&1
  echo "Finished running $script" | tee -a $log_file
  echo "----------------------------------------" >> $log_file
done

echo "All scripts have completed. Check $log_file for details."