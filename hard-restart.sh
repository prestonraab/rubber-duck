kill `ps -ae | grep python | cut -d ' ' -f 1`
git pull
poetry install
nohup poetry run python discord_bot2.py --restarted >> /tmp/duck.log &