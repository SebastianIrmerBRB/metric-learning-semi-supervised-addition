#!/usr/bin/env bash

set -Eeuo pipefail

PROGRAM_NAME=${0##*/}

remote_host='irmers@dipa.th-brandenburg.de'
remote_port=22
destination='./optuna-studies'
clean_destination=false
verbose=false

usage() {
    cat <<USAGE
Usage: $PROGRAM_NAME [options]

Downloads every optuna_study.db beneath the remote metric-learning log tree,
along with sibling study_config.json and sampler.pkl files when present.
The directory structure relative to the remote log root is preserved.

Options:
  -H, --remote-host HOST   SSH destination (default: $remote_host)
  -p, --remote-port PORT   SSH port (default: $remote_port)
  -d, --destination PATH   Extraction directory (default: $destination)
      --clean-destination  Delete the destination before extraction
  -v, --verbose            List extracted files
  -h, --help               Show this help text
USAGE
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'Warning: %s\n' "$*" >&2
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
        --clean-destination)
            clean_destination=true
            shift
            ;;
        -v|--verbose)
            verbose=true
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

[[ -n $destination ]] || fail 'Destination must not be empty.'
[[ $destination != '/' ]] || fail 'Refusing to use / as the destination.'

require_command ssh
require_command tar
require_command mktemp
require_command find

if $clean_destination && [[ -e $destination ]]; then
    $verbose && printf 'Removing existing destination: %s\n' "$destination"
    rm -rf -- "$destination"
fi

mkdir -p -- "$destination"
destination_path=$(cd -- "$destination" && pwd -P)

archive_path=$(mktemp "${TMPDIR:-/tmp}/irmers-optuna-studies.XXXXXXXX.tar.gz")
remote_script_path=$(mktemp "${TMPDIR:-/tmp}/irmers-optuna-remote.XXXXXXXX.sh")

cat > "$remote_script_path" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

log_root="$HOME/metric-learning/logs"

if [[ ! -d $log_root ]]; then
    printf 'Remote log directory does not exist: %s\n' "$log_root" >&2
    exit 2
fi

cd "$log_root"

stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT

database_count=0
config_count=0
sampler_count=0
missing_config_count=0
missing_sampler_count=0

while IFS= read -r -d '' database; do
    relative_database=${database#./}

    source_directory=$(dirname -- "$database")
    relative_directory=$(dirname -- "$relative_database")
    destination_directory="$stage/$relative_directory"

    mkdir -p -- "$destination_directory"
    cp -- "$database" "$destination_directory/optuna_study.db"
    database_count=$((database_count + 1))

    config="$source_directory/study_config.json"
    if [[ -f $config ]]; then
        cp -- "$config" "$destination_directory/study_config.json"
        config_count=$((config_count + 1))
    else
        printf 'Warning: study_config.json not found alongside %s\n' \
            "$relative_database" >&2
        missing_config_count=$((missing_config_count + 1))
    fi

    sampler="$source_directory/sampler.pkl"
    if [[ -f $sampler ]]; then
        cp -- "$sampler" "$destination_directory/sampler.pkl"
        sampler_count=$((sampler_count + 1))
    else
        printf 'Warning: sampler.pkl not found alongside %s\n' \
            "$relative_database" >&2
        missing_sampler_count=$((missing_sampler_count + 1))
    fi
done < <(
    find . \
        -type f \
        -name 'optuna_study.db' \
        -print0
)

if ((database_count == 0)); then
    printf 'No optuna_study.db files were found beneath %s.\n' \
        "$log_root" >&2
    exit 3
fi

printf 'Exporting %d optuna_study.db file(s).\n' "$database_count" >&2
printf 'Exporting %d study_config.json file(s).\n' "$config_count" >&2
printf 'Exporting %d sampler.pkl file(s).\n' "$sampler_count" >&2

if ((missing_config_count > 0)); then
    printf '%d database(s) had no sibling study_config.json file.\n' \
        "$missing_config_count" >&2
fi

if ((missing_sampler_count > 0)); then
    printf '%d database(s) had no sibling sampler.pkl file.\n' \
        "$missing_sampler_count" >&2
fi

tar -C "$stage" -czf - .
REMOTE_SCRIPT

printf 'Downloading Optuna studies from %s:%s...\n' \
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

database_files=()
config_files=()
sampler_files=()
missing_config_files=()
missing_sampler_files=()

while IFS= read -r -d '' file; do
    database_files+=("$file")
done < <(find "$destination_path" -type f -name 'optuna_study.db' -print0)

((${#database_files[@]} > 0)) ||
    fail 'The archive was extracted, but no optuna_study.db files were found.'

while IFS= read -r -d '' file; do
    config_files+=("$file")
done < <(find "$destination_path" -type f -name 'study_config.json' -print0)

while IFS= read -r -d '' file; do
    sampler_files+=("$file")
done < <(find "$destination_path" -type f -name 'sampler.pkl' -print0)

for database_file in "${database_files[@]}"; do
    directory=${database_file%/*}

    [[ -f $directory/study_config.json ]] ||
        missing_config_files+=("$database_file")

    [[ -f $directory/sampler.pkl ]] ||
        missing_sampler_files+=("$database_file")
done

printf 'Extracted %d optuna_study.db file(s).\n' "${#database_files[@]}"
printf 'Extracted %d study_config.json file(s).\n' "${#config_files[@]}"
printf 'Extracted %d sampler.pkl file(s).\n' "${#sampler_files[@]}"
printf 'Destination:\n'
printf '  %s\n' "$destination_path"

if ((${#missing_config_files[@]} > 0)); then
    warn "${#missing_config_files[@]} database file(s) had no corresponding study_config.json."
    for database_file in "${missing_config_files[@]}"; do
        warn "Missing study_config.json for: $database_file"
    done
fi

if ((${#missing_sampler_files[@]} > 0)); then
    warn "${#missing_sampler_files[@]} database file(s) had no corresponding sampler.pkl."
    for database_file in "${missing_sampler_files[@]}"; do
        warn "Missing sampler.pkl for: $database_file"
    done
fi

if $verbose; then
    printf 'Extracted database files:\n'
    printf '  %s\n' "${database_files[@]}"

    printf 'Extracted configuration files:\n'
    if ((${#config_files[@]} > 0)); then
        printf '  %s\n' "${config_files[@]}"
    fi

    printf 'Extracted sampler files:\n'
    if ((${#sampler_files[@]} > 0)); then
        printf '  %s\n' "${sampler_files[@]}"
    fi
fi
