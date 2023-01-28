# Rubber Duck Discord Bot

## Usage
The bot listens to the configured channel, currently set in the source code as `"duck-pond"`

When a user posts a message to the duck pond, the duck bot 
creates a public thread in response. 

## TODO
- [ ] Gracefully handle deleted threads
- [ ] Clean up expired threads

## Setup
- Create an OPENAI account
  - https://openai.com/
  - Get the API key and provide it as the environment variable `OPENAI_API_KEY` to the bot
- Create a new discord application
  - https://discord.com/developers/applications
  - Get the client secret and provide it as the environment variable `DISCORD_TOKEN` to the bot
  - Under "Bot"
    - Select "Message content intent"
  - Under "OAuth2"
    - Select "bot" scope
    - Permissions:
      - Send messages
      - Create public threads
      - Send messages in threads
      - Manage threads
      - Add reactions
    - Copy the generated link, paste in the URL bar of your browser, 
      and use it to add the bot to your server
  - Install `poetry`
    - https://python-poetry.org/docs/#installation
    - `curl -sSL https://install.python-poetry.org | python3 -`
  - Clone this repo
    - requires python 3.9
    - run `poetry install` 