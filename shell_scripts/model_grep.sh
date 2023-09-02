#!/bin/bash
# requires jq, grep
#
# Usage: ./model_grep.sh [-p path] [-u utilpath] <extended regex>

# number of cpus to use by default or use -j to specify
ncpu=$(nproc --all)
[ $ncpu -gt 8 ] && ncpu=8

utilpath=./
path=./
out=safetensors_db.json
while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help)
        echo "Usage: $0 [-p path] [-u utilpath]"
        echo "  -p path      path to stable-diffusion-webui"
        echo "  -u utilpath  path to safetensors_util.py"
        echo "  -j ncpu      number of cpus to use (default: $ncpu)"
        exit 0
        ;;
      -p) path="$2/"; shift 2;;
      -u) utilpath="$2/"; shift 2;;
      -j) ncpu="$2"; shift 2;;
    esac
done

if [ ! -d "${path}models/Lora/" ]; then
  echo "Error: ${path}models/Lora/ does not exist (use -p to specify path)"
  exit 1
fi

if [ ! -e "${utilpath}/safetensors_util.py" ]; then
  echo "Error: ${utilpath}/safetensors_util.py does not exist (use -u to specify path)"
  exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: $0 [-p /path/to/stable-diffusion-webui (default: ./)] <extended regex>"
    exit 1
fi

ls -1 ${path}models/Lora/*.safetensors | parallel -n 1 -j $ncpu "python ${utilpath}/safetensors_util.py metadata {} -pm 2>/dev/null |
sed -n '1b;p' | jq '.__metadata__.ss_tag_frequency' 2>/dev/null | grep -o -E '\"[^\"]*${1}[^\"]*\": [0-9]+'| sed 's~^~'{}':~p'"
