git checkout restart_duck
git pull
poetry install
kill `ps -ae | grep python | cut -d ' ' -f 1`
nohup poetry run python discord_bot2.py --restarted >> /tmp/duck.log &