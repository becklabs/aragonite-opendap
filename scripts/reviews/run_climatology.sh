#!/bin/bash
# run_inference.sh
# This script runs the aragonite inference framework on fixed dates for each year (2015-2025)
# and is resumable. For each year, if the output file (e.g., aragonite_field_2015.nc)
# already exists, that year is skipped.
#
# Usage:
#   chmod +x run_inference.sh
#   ./run_inference.sh

# Loop over the years 2015 to 2025
for year in {2017..2022}; do
    # Build a comma-separated list of dates for the year:
    # March 20, June 20, September 20, December 20
    # for month in "03" "06"; do
    for month in "11"; do
        date="${year}-${month}-15"

        output_file="data/arag/climatology/combined_2/field_${year}_${month}.nc"

        # Check if the output file already exists
        if [ -f "$output_file" ]; then
            echo "[INFO] $output_file already exists. Skipping year $year and month $month."
            continue
        fi

        echo "-------------------------------------------------------"
        echo "[INFO] Running inference for year $year on date: $date"
        echo "-------------------------------------------------------"
        
        # Run the Python inference framework
        python -m scripts.run_framework --start "$date" --end "$date" --output_nc "$output_file"
        
        # If the command fails, exit the script with a non-zero status.
        if [ $? -ne 0 ]; then
            echo "[ERROR] Inference for year $year failed. Exiting..."
            exit 1
        fi
    done

done

echo "[INFO] All inference runs completed successfully."
