kill `ps -ae | grep python | cut -d ' ' -f 1`
kill `ps -ae | grep python | cut -d ' ' -f 3`
kill `ps -ae | grep python | cut -d ' ' -f 4`
nohup poetry run python discord_bot2.py >> /tmp/duck.log &