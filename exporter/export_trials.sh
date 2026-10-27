#!/usr/bin/env bash

set -Eeuo pipefail

PROGRAM_NAME=${0##*/}

remote_host='irmers@dipa.th-brandenburg.de'
remote_port=22
destination='./trials-files'
test_results_file_name='test-results.csv'
clean_destination=false

usage() {
    cat <<USAGE
Usage: $PROGRAM_NAME [options]

Downloads and flattens remote trials.csv files, runs aggregate_test_results.py
on the remote host, and extracts the resulting archive locally.

Options:
  -H, --remote-host HOST       SSH destination (default: $remote_host)
  -p, --remote-port PORT       SSH port (default: $remote_port)
  -d, --destination PATH       Extraction directory (default: $destination)
  -o, --test-results-file NAME Aggregated CSV filename (default: $test_results_file_name)
      --clean-destination      Delete the destination before extraction
  -h, --help                   Show this help text
USAGE
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "$1 was not found."
}

cleanup() {
    if [[ -n ${archive_path:-} && -e $archive_path ]]; then
        rm -f -- "$archive_path"
    fi

    if [[ -n ${remote_script_path:-} && -e $remote_script_path ]]; then
        rm -f -- "$remote_script_path"
    fi
}

trap cleanup EXIT
trap 'printf "Error: command failed on line %d.\n" "$LINENO" >&2' ERR

while (($# > 0)); do
    case $1 in
        -H|--remote-host)
            (($# >= 2)) || fail "$1 requires a value."
            remote_host=$2
            shift 2
            ;;
        -p|--remote-port)
            (($# >= 2)) || fail "$1 requires a value."
            remote_port=$2
            shift 2
            ;;
        -d|--destination)
            (($# >= 2)) || fail "$1 requires a value."
            destination=$2
            shift 2
            ;;
        -o|--test-results-file)
            (($# >= 2)) || fail "$1 requires a value."
            test_results_file_name=$2
            shift 2
            ;;
        --clean-destination)
            clean_destination=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            (($# == 0)) || fail "Unexpected positional argument: $1"
            ;;
        -* )
            fail "Unknown option: $1"
            ;;
        *)
            fail "Unexpected positional argument: $1"
            ;;
    esac
done

[[ $remote_host =~ ^[A-Za-z0-9_][A-Za-z0-9_.@:-]*$ ]] ||
    fail "Remote host contains unsupported characters: $remote_host"

[[ $remote_port =~ ^[0-9]+$ ]] || fail 'Remote port must be an integer.'
((remote_port >= 1 && remote_port <= 65535)) ||
    fail 'Remote port must be between 1 and 65535.'

[[ $test_results_file_name =~ ^[A-Za-z0-9._-]+\.csv$ ]] ||
    fail 'Test-results filename must be a simple .csv filename without a directory.'

[[ -n $destination ]] || fail 'Destination must not be empty.'
[[ $destination != '/' ]] || fail 'Refusing to use / as the destination.'

require_command ssh
require_command tar
require_command base64
require_command mktemp
require_command find

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
aggregator_path="$script_directory/aggregate_test_results.py"
[[ -f $aggregator_path ]] || fail "The test-result aggregator was not found: $aggregator_path"

if $clean_destination && [[ -e $destination ]]; then
    rm -rf -- "$destination"
fi

mkdir -p -- "$destination"
destination_path=$(cd -- "$destination" && pwd -P)

archive_path=$(mktemp "${TMPDIR:-/tmp}/irmers-trials.XXXXXXXX.tar.gz")
remote_script_path=$(mktemp "${TMPDIR:-/tmp}/irmers-trials-remote.XXXXXXXX.sh")

encoded_aggregator=$(base64 < "$aggregator_path" | tr -d '\r\n')

{
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'aggregator_base64=%q\n' "$encoded_aggregator"
    printf 'test_results_file=%q\n' "$test_results_file_name"

    cat <<'REMOTE_SCRIPT'

log_root="$HOME/metric-learning/logs"

if [[ ! -d $log_root ]]; then
    printf 'Remote log directory does not exist: %s\n' "$log_root" >&2
    exit 2
fi

cd "$log_root"

stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT

count=0

while IFS= read -r -d '' file; do
    # Example:
    # ./cifar/config/hpo/experiment/trial_0000/cv_date/trials.csv
    relative_path=${file#./}
    parent_path=${relative_path%/trials.csv}

    # Preserve only the top-level directory.
    base_directory=${parent_path%%/*}

    # Flatten everything below the top-level directory.
    remaining_path=${parent_path#*/}
    output_name=${remaining_path//\//__}.csv

    destination_directory="$stage/$base_directory"
    destination_file="$destination_directory/$output_name"

    mkdir -p -- "$destination_directory"

    if [[ -e $destination_file ]]; then
        printf 'Duplicate generated path: %s/%s\n' \
            "$base_directory" "$output_name" >&2
        exit 4
    fi

    cp -- "$file" "$destination_file"
    count=$((count + 1))
done < <(
    find . \
        -type f \
        -name 'trials.csv' \
        -print0
)

if ((count == 0)); then
    printf 'No trials.csv files were found beneath %s.\n' "$log_root" >&2
    exit 3
fi

printf 'Exporting %d trials.csv files.\n' "$count" >&2

aggregator="$stage/.aggregate_test_results.py"

if base64 --decode </dev/null >/dev/null 2>&1; then
    printf '%s' "$aggregator_base64" | base64 --decode > "$aggregator"
elif base64 -d </dev/null >/dev/null 2>&1; then
    printf '%s' "$aggregator_base64" | base64 -d > "$aggregator"
else
    printf 'The remote base64 command does not support decoding.\n' >&2
    exit 6
fi

if command -v python3 >/dev/null 2>&1; then
    python_bin=$(command -v python3)
elif [[ -x $HOME/metric-learning/.venv/bin/python ]]; then
    python_bin="$HOME/metric-learning/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    python_bin=$(command -v python)
else
    printf 'Python was not found on the remote host.\n' >&2
    exit 5
fi

"$python_bin" "$aggregator" \
    --log-root "$PWD" \
    --output "$stage/$test_results_file"

rm -- "$aggregator"
tar -C "$stage" -czf - .
REMOTE_SCRIPT
} > "$remote_script_path"

printf 'Downloading trials and aggregated test results from %s:%s...\n' \
    "$remote_host" "$remote_port"

if ssh -p "$remote_port" "$remote_host" bash -p -s \
    < "$remote_script_path" > "$archive_path"; then
    :
else
    status=$?
    fail "SSH export failed with exit code $status."
fi

[[ -f $archive_path ]] || fail 'SSH completed, but no archive was created.'
[[ -s $archive_path ]] || fail 'SSH completed, but the resulting archive is empty.'

tar -xzf "$archive_path" -C "$destination_path"

test_results_path="$destination_path/$test_results_file_name"
[[ -f $test_results_path ]] ||
    fail "The remote export did not contain $test_results_file_name."

trial_csv_count=0
while IFS= read -r -d '' csv_file; do
    if [[ $csv_file != "$test_results_path" ]]; then
        trial_csv_count=$((trial_csv_count + 1))
    fi
done < <(find "$destination_path" -type f -name '*.csv' -print0)

if command -v python3 >/dev/null 2>&1; then
    local_python=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
    local_python=$(command -v python)
else
    local_python=''
fi

if [[ -n $local_python ]]; then
    test_result_count=$(
        "$local_python" - "$test_results_path" <<'PYTHON'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PYTHON
    )
else
    # Fallback for systems without local Python. This assumes no embedded
    # newlines inside quoted CSV fields.
    test_result_count=$(awk 'NR > 1 { count++ } END { print count + 0 }' \
        "$test_results_path")
fi

printf 'Extracted %d trial CSV file(s) to:\n' "$trial_csv_count"
printf '  %s\n' "$destination_path"
printf 'Aggregated %d final test result(s) to:\n' "$test_result_count"
printf '  %s\n' "$test_results_path"
