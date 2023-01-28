# the os module helps us access environment variables
# i.e., our API keys
import os
import logging
from datetime import datetime

from discord import ChannelType

logging.basicConfig(level=logging.DEBUG)

# the Discord Python API
import discord
import openai


def load_env():
    with open('secrets.env') as file:
        for line in file:
            key, value = line.strip().split('=')
            os.environ[key] = value


load_env()
openai.api_key = os.environ['OPENAI_API_KEY']

AI_ENGINE = 'text-davinci-003'
PREFIX = """
RD is a friendly computer science TA for an introductory python course. 
RD asks questions to help them learn how to solve the problem.
RD avoids giving the student the answer directly.
RD first makes sure the student understands the problem.
Then RD asks questions to help the student decompose the problem into smaller subproblems,
suggesting the student use functions to abstract away the details of the subproblems.
RD essentially acts as a sounding board for the student.
If the student is not able to solve the problem, RD suggests they talk to a human TA.
S is a student looking for help.
S: 
"""

CONVERSATION_TIMEOUT = 60 * 3  # three minutes


class MyClient(discord.Client):
    def __init__(self, listenning_channel):
        # adding intents module to prevent intents error in __init__ method in newer versions of Discord.py
        intents = discord.Intents.default()  # Select all the intents in your bot settings as it's easier
        intents.message_content = True
        super().__init__(intents=intents)
        self.conversations = {}
        self.timestamps = {}
        self.listenning_channel = listenning_channel

    def query(self, thread_id, timestamp: datetime, text):
        """
        Query the OPENAI API
        """
        # if the user is not in the conversation dictionary
        # or it has been a while since the last message
        # start a new conversation
        if thread_id not in self.conversations or \
                (timestamp - self.timestamps[thread_id]).total_seconds() > CONVERSATION_TIMEOUT:
            self.conversations[thread_id] = PREFIX
            self.timestamps[thread_id] = timestamp

        logging.debug(f"User said: {text}")

        self.conversations[thread_id] += text + '\nRD: '

        completion = openai.Completion.create(
            model=AI_ENGINE,
            prompt=self.conversations[thread_id],
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

        self.conversations[thread_id] += response + '\nS: '
        return response

    async def on_ready(self):
        # print out information when the bot wakes up
        logging.info('Logged in as')
        logging.info(self.user.name)
        logging.info(self.user.id)
        logging.info('------')

    async def on_message(self, message):
        """
        This function is called whenever the bot sees a message in a channel
        If the message is a message in the self.listenning channel,
          the bot creates a thread in response to that message
        If the message is in a thread under the self.listenning channel,
          the bot continues the conversation in that thread
        The bot ignores all other messages.
        """
        # ignore messages from the bot itself
        if message.author == self.user:
            return

        # if the message is in the self.listenning channel
        # and the message is not a thread
        if message.channel.name == self.listenning_channel:
            # create a thread in response to the message
            await self.create_conversation(message)

        # if the message is in a thread under the self.listenning channel
        elif message.channel.id in self.conversations:
            # continue the conversation
            await self.continue_conversation(message.channel, datetime.utcnow(), message.content)

        # otherwise, ignore the message
        else:
            return

    async def create_conversation(self, message):
        """
        Create a thread in response to this message.
        """
        # get the channel from the message
        channel = message.channel

        # create a public thread in response to the message
        thread = await channel.create_thread(
            name=message.content,
            type=ChannelType.public_thread,
            auto_archive_duration=60
        )

        # add the thread to the conversations dictionary
        self.conversations[thread.id] = PREFIX
        self.timestamps[thread.id] = datetime.utcnow()

        # continue the conversation
        await self.continue_conversation(
            thread, self.timestamps[thread.id], self.conversations[thread.id], user=message.author)

    async def continue_conversation(
            self, thread, timestamp: datetime, content: str, user=None):
        """
        Use the OPNENAI API to continue the conversation
        """
        # while the bot is waiting on a response from the model
        # set its status as typing for user-friendliness
        async with thread.typing():
            response = self.query(thread.id, timestamp, content)

        if not response:
            response = 'RubberDuck encountered an error.'

        # mention the user at the beginning of the response
        if user is not None:
            response = f"{user.mention} {response}"
        # send the model's response to the Discord channel
        await thread.send(response)


def main():
    client = MyClient('duck-pond')
    client.run(os.environ['DISCORD_TOKEN'])


if __name__ == '__main__':
    main()
