import json
import os
import logging
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import shlex

from typing import TypedDict

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

AI_ENGINE = 'gpt-4'
CONVERSATION_TIMEOUT = 60 * 3  # three minutes


class GPTMessage(TypedDict):
    role: str
    content: str


@dataclass
class Conversation:
    thread: discord.Thread
    guild_id: int
    thread_id: int
    thread_name: str
    started_by: str
    first_message: datetime
    last_message: datetime
    messages: list[GPTMessage]

    def to_json(self):
        return {
            "guild_id": self.guild_id,
            "thread_id": self.thread_id,
            "thread_name": self.thread_name,
            "started_by": self.started_by,
            "first_message": self.first_message.isoformat(),
            "last_message": self.last_message.isoformat(),
            "messages": self.messages
        }

    @staticmethod
    def from_json(jobj: dict, thread: discord.Thread) -> 'Conversation':
        jobj['first_message'] = datetime.fromisoformat(jobj['first_message'])
        jobj['last_message'] = datetime.fromisoformat(jobj['last_message'])
        jobj['thread'] = thread
        return Conversation(**jobj)


class MyClient(discord.Client):
    def __init__(self, prompt_dir: Path, conversation_dir: Path):
        # adding intents module to prevent intents error in __init__ method in newer versions of Discord.py
        intents = discord.Intents.default()  # Select all the intents in your bot settings as it's easier
        intents.message_content = True
        super().__init__(intents=intents)

        self._load_prompts(prompt_dir)
        self.conversation_dir = conversation_dir
        self.conversations = {}
        self.guild_dict = {}  # Loaded in on_ready

    def _load_prompts(self, prompt_dir: Path):
        self.prompts = {}
        for file in prompt_dir.glob("**/*"):
            if file.suffix == '.txt':
                self.prompts[file.stem] = file.read_text()

    def __enter__(self):
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        # serialize the conversations
        logging.info('Serializing conversations')
        for conversation in self.conversations.values():
            self._serialize_conversation(conversation)
        logging.info('Done serializing conversations')

    def _handle_interrupt(self, signum=None, frame=None):
        self.__exit__()
        exit()

    def _serialize_conversation(self, conversation: Conversation):
        # Save conversation as JSON in self.conversations_dir
        logging.debug(f'Serializing conversation {conversation.thread_id}')
        filename = f'{conversation.guild_id}_{conversation.thread_id}.json'
        with open(self.conversation_dir / filename, 'w') as file:
            json.dump(conversation.to_json(), file)

    def _load_conversation(self, filename: str):
        # Load conversation from JSON in self.conversations_dir
        logging.debug(f'Loading conversation {filename}')
        try:
            with open(self.conversation_dir / filename) as file:
                jobj = json.load(file)

            guild = self.guild_dict.get(jobj['guild_id'])
            if guild is None:
                return
            thread_id = jobj['thread_id']
            thread = self.get_channel(thread_id)
            self.conversations[thread_id] = Conversation.from_json(jobj, thread)
        except Exception as ex:
            logging.exception(f"Unable to load conversation: {filename}")

    async def query(self, conversation: Conversation, message_text: str):
        """
        Query the OPENAI API
        """
        logging.debug(f"User said: {message_text}")

        conversation.messages.append(dict(role='user', content=message_text))

        completion = await openai.ChatCompletion.acreate(
            model=AI_ENGINE,
            messages=conversation.messages
        )
        logging.debug(f"Completion: {completion}")

        response_message = completion.choices[0]['message']
        response = response_message['content'].strip()
        logging.debug(f"Response: {response}")

        conversation.messages.append(response_message)

        return response

    async def on_ready(self):
        self.guild_dict = {guild.id: guild async for guild in self.fetch_guilds(limit=150)}

        # Load conversations from JSON in self.conversations_dir
        logging.info('Loading conversations')
        for file in self.conversation_dir.iterdir():
            if file.suffix == '.json':
                self._load_conversation(file.name)
        logging.info('Done loading conversations')

        # print out information when the bot wakes up
        logging.info('Logged in as')
        logging.info(self.user.name)
        logging.info(self.user.id)
        logging.info('------')
        channels = self.get_all_channels()
        for channel in channels:
            if channel.name == 'control-duck':
                await channel.send('Duck online')

    async def restart(self, message):
        """
        Restart the bot
        :param message: The message that triggered the restart
        """
        await message.channel.send(f'Restart requested.')
        os.chdir(Path(__file__).parent)
        await self.execute_command('git fetch', message.channel)
        await self.execute_command('git reset --hard', message.channel)
        await self.execute_command('git clean -f', message.channel)
        await self.execute_command('git pull --rebase=false', message.channel)
        await self.execute_command('poetry install', message.channel)
        await message.channel.send(f'Restarting.')
        subprocess.Popen(["bash", "restart.sh"])
        return

    async def say_in_channel(self, channel, text, prefix="", suffix=""):
        """
        Say something in a channel
        :param channel: The channel to say something in
        :param output: The message to say, long messages will be split into multiple messages
        """
        # Async methods like for loops better than while loops
        # Split output into 1800 character chunks
        for i in range(len(text) // 1800):
            await channel.send(f'{prefix}{text[:1800]}{suffix}')
            text = text[1800:]

        await channel.send(f'{prefix}{text}{suffix}')

    async def execute_command(self, text, channel):
        """
        Execute a command in the shell and return the output to the channel
        """
        # Run command using shell and pipe output to channel
        await self.say_in_channel(channel, text, "```ps\n$ ", "```")
        process = subprocess.run(text, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Get output of command and send to channel
        errors = process.stderr.decode('utf-8')
        if errors:
            await self.say_in_channel(channel, errors, prefix=f'Errors: ```', suffix='```')
        output = str(process.stdout.decode('utf-8'))
        if output:
            await self.say_in_channel(channel, output, prefix=f'```', suffix='```')
        if len(output) > 2000:
            await channel.send(f'Done.')
        return

    async def execute_message(self, message):
        """
        Execute a command in the shell and return the output to the channel
        Removes the starting !
        :param message: discord.Message object, used to get the command
        """
        await self.execute_command(message.content[1:], message.channel)
        return

    async def display_help(self, message):
        await message.channel.send(
            "!restart - restart the bot\n"
            "!log - print the log file\n"
            "!rmlog - remove the log file\n"
            "!status - print a status message\n"
            "!help - print this message\n"
        )

    async def control_on_message(self, message):
        """
        This function is called whenever the bot sees a message in a control channel
        :param message:
        :return:
        """
        content = message.content
        if content.startswith('!restart'):
            await self.restart(message)

        elif content.startswith('!log'):
            await self.execute_command("cat /tmp/duck.log", message.channel)
            await message.channel.send('Done.')

        elif content.startswith('!rmlog'):
            await self.execute_command("rm /tmp/duck.log", message.channel)
            await self.execute_command("touch /tmp/duck.log", message.channel)
            await message.channel.send('Done.')

        elif content.startswith('!status'):
            await message.channel.send('I am alive.')

        elif content.startswith('!help'):
            await self.display_help(message)
        elif content.startswith('!'):
            await message.channel.send('Unknown command. Try !help')

    async def on_message(self, message: discord.Message):
        """
        This function is called whenever the bot sees a message in a channel
        If the message is in a listen channel
          the bot creates a thread in response to that message
        If the message is in a conversation thread,
          the bot continues the conversation in that thread
        The bot ignores all other messages.
        """
        # ignore messages from the bot itself
        if message.author == self.user:
            return

        if message.content.startswith('//'):
            return

        if message.channel.name == 'control-duck':
            await self.control_on_message(message)
            return

        if message.content.startswith('!help'):
            await message.channel.send(
                "```Possible duck channel commands:\n" +
                "\n".join(["\t!" + key for key in self.prompts.keys()]) +
                "```")
            return

        # if the message is in a listen channel, create a thread
        # if the user provides a prompt, use that instead
        # ignore prompt injections for gpt4 channel
        if message.channel.name in self.prompts:
            if not message.channel.name == "gpt4":
                prefix = self.prompts["duck-pond"]
                if message.content.startswith('!'):
                    assignment = message.content.strip('!')
                    
                    # Bit prompt injection
                    if "1" in assignment or "2" in assignment and assignment in self.prompts:
                        prefix += self.prompts["bit-prompt"] + self.prompts[assignment]
                    elif assignment in self.prompts:
                        prefix += self.prompts[assignment]

            else:
                prefix = self.prompts["gpt4"]
            await self.create_conversation(prefix, message)

        # if the message is in an active thread, continue the conversation
        elif message.channel.id in self.conversations:
            await self.continue_conversation(
                self.conversations[message.channel.id], message.content)

        # otherwise, ignore the message
        else:
            return

    async def create_conversation(self, prefix, message):
        """
        Create a thread in response to this message.
        """
        # get the channel from the message
        channel = message.channel

        # create a public thread in response to the message
        thread = await channel.create_thread(
            name=message.content[:20],
            type=ChannelType.public_thread,
            auto_archive_duration=60
        )
        welcome = f'{message.author.mention} What can I do for you?'

        conversation = Conversation(
            guild_id=thread.guild.id,
            thread=thread,
            thread_id=thread.id,
            thread_name=thread.name,
            started_by=message.author.name,
            first_message=datetime.utcnow(),
            last_message=datetime.utcnow(),
            messages=[
                dict(role='system', content=prefix or message.content),
                dict(role='assistant', content=welcome)
            ]
        )
        self.conversations[thread.id] = conversation
        async with thread.typing():
            await thread.send(welcome)

    def parse_blocks(self, text: str, limit=2000):
        tick = '`'
        block = ""
        current_fence = ""
        for line in text.splitlines():
            if len(block) + len(line) > limit - 3:
                if block:
                    if current_fence:
                        block += '```'
                    yield block
                    block = current_fence

            block += ('\n' + line) if block else line

            if line.strip().startswith(tick * 3):
                if current_fence:
                    current_fence = ""
                else:
                    current_fence = line

        if block:
            yield block

    async def send(self, thread: discord.Thread, text: str):
        for block in self.parse_blocks(text):
            await thread.send(block)

    async def continue_conversation(
            self, conversation: Conversation, message_text: str):
        """
        Use the OPNENAI API to continue the conversation
        """
        thread = conversation.thread

        # while the bot is waiting on a response from the model
        # set its status as typing for user-friendliness
        async with thread.typing():
            response = await self.query(conversation, message_text)

            if not response:
                response = 'RubberDuck encountered an error.'

            # send the model's response to the Discord channel
            await self.send(thread, response)


def main(prompts: Path, conversations: Path):
    with MyClient(prompts, conversations) as client:
        client.run(os.environ['DISCORD_TOKEN'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=Path, default='prompts')
    parser.add_argument('--conversations', type=Path, default='conversations')
    args = parser.parse_args()
    main(args.prompts, args.conversations)
