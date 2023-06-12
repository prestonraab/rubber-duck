kill `ps -ae | grep -v grep | grep python | awk '{$1=$1};1' | cut -d ' ' -f 1`
nohup poetry run python discord_bot3.py >> /tmp/duck.log &