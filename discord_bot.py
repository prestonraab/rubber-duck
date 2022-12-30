# the os module helps us access environment variables
# i.e., our API keys
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

# the Discord Python API
import discord
import openai

AI_ENGINE = 'davinci'


def load_env():
    with open('secrets.env') as file:
        for line in file:
            key, value = line.strip().split('=')
            os.environ[key] = value


load_env()
openai.api_key = os.environ['OPENAI_API_KEY']

PREFIX = "RD is a friendly computer science TA for an introductory python course. " \
         "S is a student looking for help.\nS: "

CONVERSATION_TIMEOUT = 60 * 3  # three minutes


class MyClient(discord.Client):
    def __init__(self):
        # adding intents module to prevent intents error in __init__ method in newer versions of Discord.py
        intents = discord.Intents.default()  # Select all the intents in your bot settings as it's easier
        intents.message_content = True
        super().__init__(intents=intents)
        self.conversations = {}
        self.timestamps = {}

    def query(self, user, timestamp: datetime, text):
        """
        Query the OPENAI API
        """
        # if the user is not in the conversation dictionary
        # or it has been a while since the last message
        # start a new conversation
        if user not in self.conversations or (timestamp - self.timestamps[user]) > CONVERSATION_TIMEOUT:
            self.conversations[user] = PREFIX
            self.timestamps[user] = timestamp

        logging.debug(f"User said: {text}")

        self.conversations[user] += text + '\nRD: '

        completion = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.conversations[user],
            temperature=0.9,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=["S: ", "RD: "],
        )
        logging.debug(f"Completion: {completion}")

        response = completion.choices[0].text.strip()
        logging.debug(f"Response: {response}")

        self.conversations[user] += response + '\nS: '
        return response

    async def on_ready(self):
        # print out information when the bot wakes up
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        """
        this function is called whenever the bot sees a message in a channel
        """
        # check if the message is a direct message to the bot
        if message.guild is not None:
            return

        # ignore the message if it comes from the bot itself
        if message.author.id == self.user.id:
            return

        # Get the user from the message
        user = message.author

        # Get a DM channel for the user
        dm_channel = await user.create_dm()

        # Get the timestamp of the message
        timestamp = message.created_at.timestamp()

        # while the bot is waiting on a response from the model
        # set its status as typing for user-friendliness
        async with dm_channel.typing():
            response = self.query(user.id, timestamp, message.content)

        if not response:
            response = 'RubberDuck encountered an error.'

        # send the model's response to the Discord channel
        await dm_channel.send(response)


def main():
    client = MyClient()
    client.run(os.environ['DISCORD_TOKEN'])


if __name__ == '__main__':
    main()
