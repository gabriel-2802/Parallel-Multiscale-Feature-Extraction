#!/bin/bash

MAIN_DIR=$1      # ex: project1
EXEC=$2          # ex: myprog

if [[ -z "$MAIN_DIR" || -z "$EXEC" ]]; then
    echo "Usage: ./remote_run.sh <main_dir> <exec_name>"
    exit 1
fi

SCRIPT_NAME="vtune_run.sh"

echo "Building executable $EXEC inside $MAIN_DIR..."

cd "$MAIN_DIR" || { echo "Directory not found"; exit 1; }

# compile using the Makefile inside main_dir
make EXEC=$EXEC

echo "Creating VTune job script..."

cat <<EOF > $SCRIPT_NAME
#!/bin/bash
source /opt/intel/oneapi/vtune/latest/vtune-vars.sh

vtune -collect performance-snapshot ./$EXEC
vtune -collect hotspots ./$EXEC
EOF

chmod +x $SCRIPT_NAME

echo "Submitting job..."

JOB_SUBMIT_OUTPUT=$(sbatch --time 00:10:00 -p xl ./$SCRIPT_NAME)
echo "$JOB_SUBMIT_OUTPUT"

# Extract JobID
JOBID=$(echo $JOB_SUBMIT_OUTPUT | awk '{print $4}')
echo "Submitted job with ID: $JOBID"

# Wait for job to finish before cleaning
echo "Waiting for job to finish..."
while squeue -j $JOBID >/dev/null 2>&1; do
    sleep 5
done

echo "Job finished."

echo "Cleaning files on FEP (except r* result folders)..."
find . -maxdepth 1 ! -name "r*ps" ! -name "r*hs" ! -name "." -exec rm -rf {} \;

echo "Remote cleanup done."
