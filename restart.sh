kill "$(pgrep -a python | awk '{$1=$1};1' | cut -d ' ' -f 1)"
nohup poetry run python discord_bot2.py >> /tmp/duck.log &