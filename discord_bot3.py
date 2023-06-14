import datetime
import json
import logging
import os
import signal
import subprocess
from pathlib import Path

import discord
from discord import ChannelType
import openai
import argparse

from typing import Callable, TypedDict, Union

from quest import event, signal as quest_signal
from quest.workflow_manager import WorkflowSerializer, WorkflowManager
from quest.json_seralizers import JsonMetadataSerializer, JsonEventSerializer
from quest.workflow import WorkflowFunction

logging.basicConfig(level=logging.DEBUG)
INPUT_EVENT_NAME = 'get_input'
CHAT_EVENT_NAME = 'query'


def load_env():
    with open('.env') as f:
        for line in f:
            key, value = line.strip().split('=')
            os.environ[key] = value


load_env()
openai.api_key = os.environ['OPENAI_API_KEY']

AI_ENGINE = 'gpt-4'
CONVERSATION_TIMEOUT = 60 * 3  # three minutes


class GPTMessage(TypedDict):
    role: str
    content: str


categories = {
    "Chat": ["chat", "gpt", "gpt-4", "duck", "talk", "discuss", "help", "assistance", "need"],
    "Greeting": ["hi", "hello", "hey", "howdy", "greetings", "good morning", "good evening", "good afternoon"],
    "Goodbye": ["bye", "goodbye", "goodnight", "see you later", "see ya", "cya"],
    "Gratitude": ["thanks", "thank you"],
    "Affirmation": ["yes", "yep", "yeah", "indeed", "that's right", "ok", "okay", "cool", "great", "sounds good"],
    "Negative": ["no", "nope", "nah", "that's wrong", "that's incorrect", "that is wrong", "that is incorrect"],
    "Apology": ["sorry", "my apologies", "my apologies", "my bad"],
    "Condescension": ["whatever", "sure", "I guess", "if you say so", "yeah, right"],
    "Agreement": ["I agree", "I disagree", "I don't agree", "I don't disagree"],
    "Question": ["what", "why", "how", "when", "where", "who"],
    "Clarification": ["what do you mean", "what do you mean?", "what does that mean", "what does that mean?",
                      "what are you talking about", "what are you talking about?", "what are you saying",
                      "what are you saying?",
                      "what are you trying to say", "what are you trying to say?", "what do you mean by that"],
    "Exclamation": ["wow", "wow!", "wow...", "oh", "oh!", "oh...", "huh", "huh?", "huh...", "ah"],
    "Hedge": ["maybe", "perhaps", "I don't know", "I don't think so"]
}


async def categorize(text: str):
    for category, keywords in categories.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            return category


class DuckResponseFlow:
    def __init__(self, thread, control_channels: list[discord.TextChannel]):
        self.thread = thread
        self.chat_messages: list[GPTMessage] = []
        self.control_channels = control_channels
        self.start_time = datetime.datetime.now()

    async def prompt(self, prompt: str):
        await self.display(prompt)
        return await self.get_input()

    @event
    async def display(self, text: str):
        async with self.thread.typing():
            await send(self.thread, text)

    async def category_chat(self, time_left: int):
        user_response = await self.get_input()
        category = await categorize(user_response)
        if category == "Chat":
            await self.respond(user_response)
        else:
            await self.display(f"Category: {category}. Time left: {time_left}. Enter your response: ")

    async def __call__(self):
        user_response = await self.prompt("How can I help you?")

        while True:
            async with self.thread.typing():
                response = await self.query(user_response)
                if not response:
                    response = 'RubberDuck encountered an error.'

            user_response = await self.prompt(f"Chat said {response}")

        # await self.thread.send("All events completed.")

    @quest_signal(INPUT_EVENT_NAME)
    def get_input(self):
        ...

    @event
    async def query(self, message_text: str):
        """
        Query the OPENAI API
        """
        self.chat_messages.append(dict(role='user', content=message_text))

        completion = await openai.ChatCompletion.acreate(
            model=AI_ENGINE,
            messages=self.chat_messages
        )
        logging.debug(f"Completion: {completion}")

        response_message = completion.choices[0]['message']
        response = response_message['content'].strip()
        logging.debug(f"Response: {response}")

        self.chat_messages.append(response_message)

        return response

    @event
    async def display_control(self, text: str):
        for channel in self.control_channels:
            await channel.send(text)

    async def respond(self, message_text: str):

        """
        Use the OPNENAI API to continue the conversation
        """
        # while the bot is waiting on a response from the model
        # set its status as typing for user-friendliness
        async with self.thread.typing():
            response = await self.query(self.chat_messages, message_text)

            if not response:
                response = 'RubberDuck encountered an error.'

            # send the model's response to the Discord channel
            #await self.display(response)
            await send(self.thread, response)




def parse_blocks(text: str, limit=2000):
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


async def send(thread: Union[discord.Thread, discord.TextChannel], text: str):
    for block in parse_blocks(text):
        await thread.send(block)


async def display_help(message):
    await message.channel.send(
        "!restart - restart the bot\n"
        "!log - print the log file\n"
        "!rm log - remove the log file\n"
        "!status - print a status message\n"
        "!help - print this message\n"
    )


async def execute_command(text, channel):
    """
    Execute a command in the shell and return the output to the channel
    """
    # Run command using shell and pipe output to channel
    work_dir = Path(__file__).parent
    await send(channel, f"```ps\n$ {text}```")
    process = subprocess.run(text, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir)
    # Get output of command and send to channel
    errors = process.stderr.decode('utf-8')
    if errors:
        await send(channel, f'Error: ```{errors}```')
    output = process.stdout.decode('utf-8')
    if output:
        await send(channel, f'```{output}```')
    return


async def restart(message):
    """
    Restart the bot
    :param message: The message that triggered the restart
    """
    await message.channel.send(f'Restart requested.')
    await execute_command('git fetch', message.channel)
    await execute_command('git reset --hard', message.channel)
    await execute_command('git clean -f', message.channel)
    await execute_command('git pull --rebase=false', message.channel)
    await execute_command('poetry install', message.channel)
    await message.channel.send(f'Restarting.')
    subprocess.Popen(["bash", "restart.sh"])
    return


async def control_on_message(message, log_file: Path):
    """
    This function is called whenever the bot sees a message in a control channel
    :param log_file:
    :param message:
    :return:
    """
    content = message.content
    if content.startswith('!restart'):
        await restart(message)

    elif content.startswith('!log'):
        await message.channel.send(file=discord.File(log_file))

    elif content.startswith('!rmlog') or content.startswith('!rm log'):
        await execute_command("rm " + str(log_file), message.channel)
        await execute_command("touch " + str(log_file), message.channel)

    elif content.startswith('!status'):
        await message.channel.send('🦆')

    elif content.startswith('!help'):
        await display_help(message)
    elif content.startswith('!'):
        await message.channel.send('Unknown command. Try !help')


class DiscordWorkflowSerializer(WorkflowSerializer):
    def __init__(self, create_workflow: Callable[[], WorkflowFunction], discord_client: discord.Client, folder: Path):
        self.create_workflow = create_workflow
        self.folder = folder
        self.discord_client = discord_client

    def serialize_workflow(self, workflow_id: str, workflow: WorkflowFunction):
        # Serialize workflow to specified folder location with metadata
        # create a dict to serialize
        metadata = {"tid": workflow_id, "wid": workflow_id}
        file_to_save = "workflow" + workflow_id + ".json"
        with open(self.folder / file_to_save, 'w') as file:
            json.dump(metadata, file)

    def deserialize_workflow(self, workflow_id: str) -> WorkflowFunction:
        # Load the file with key
        file_to_load = "workflow" + workflow_id + ".json"
        with open(self.folder / file_to_load) as file:
            workflow_metadata = json.load(file)
            thread = self.get_thread(workflow_metadata['tid'])
            return self.create_workflow(thread)

    def get_thread(self, tid) -> discord.Thread:
        thread = self.discord_client.get_channel(int(tid))
        return thread


class MyClient(discord.Client):
    def __init__(self, prompt_dir: Path, log_file: Path):
        # adding intents module to prevent intents error in __init__ method in newer versions of Discord.py
        intents = discord.Intents.default()  # Select all the intents in your bot settings as it's easier
        intents.message_content = True
        super().__init__(intents=intents)

        self.workflow_manager = None
        self.log_file = log_file

        self._load_prompts(prompt_dir)
        self.guild_dict = {}  # Loaded in on_ready

    def _load_prompts(self, prompt_dir: Path):
        self.prompts = {}
        for file in prompt_dir.iterdir():
            if file.suffix == '.txt':
                self.prompts[file.stem] = file.read_text()

    def __enter__(self):
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the state is saved in workflow manager
        self.workflow_manager.save_workflows()

    def _handle_interrupt(self, signum=None, frame=None):
        self.__exit__(None, None, None)
        exit()

    async def on_ready(self):
        self.guild_dict = {guild.id: guild async for guild in self.fetch_guilds(limit=150)}

        self._load_control_channels()
        # contextualize members
        self.workflow_manager.load_workflows()
        await self.workflow_manager.resume_workflows()
        # print out information when the bot wakes up
        logging.info('Logged in as')
        logging.info(self.user.name)
        logging.info(self.user.id)
        logging.info('------')
        for channel in self.control_channels:
            await channel.send('Duck online')

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

        if message.channel.id in self.control_channel_ids:
            await control_on_message(message, self.log_file)
            return

        # if the message is in a listen channel, create a thread
        if message.channel.name in self.prompts:
            await self._create_conversation(self.prompts[message.channel.name], message)

        # if the message is in an active thread, continue the conversation
        elif self.workflow_manager.has_workflow(str(message.channel.id)):
            await self._continue_conversation(str(message.channel.id), message.content)

        # otherwise, ignore the message
        else:
            return

    async def _create_conversation(self, prefix, message: discord.Message):
        # Create a private thread in the message channel
        thread = await message.channel.create_thread(
            name=message.author.name + ": " + message.content[:20],
            type=ChannelType.public_thread,
            auto_archive_duration=60
        )

        messages = [
            dict(role='system', content=prefix or message.content)
        ]
        await self.workflow_manager.start_async_workflow(
            str(thread.id),
            DuckResponseFlow(thread),
            messages,
            self.control_channels)
        # TODO::Should we handle messages that finish out of the gate?

    async def _continue_conversation(self, thread_id, text: str):
        # Get the conversation
        result = await self.workflow_manager.signal_async_workflow(thread_id, INPUT_EVENT_NAME, text)
        if result is not None:
            logging.debug(f'Conversation {thread_id} complete.')

    def set_workflow_manager(self, workflow_manager: WorkflowManager):
        self.workflow_manager = workflow_manager

    def _load_control_channels(self):
        with open('config.json') as file:
            config = json.load(file)
        self.control_channel_ids = config['control_channels']
        print(self.control_channel_ids)
        self.control_channels = [c for c in self.get_all_channels() if c.id in self.control_channel_ids]


def main(prompts: Path, log_file: Path):
    # create client
    client = MyClient(prompts, log_file)
    # set path for saved state and init workflow manager
    saved_state = Path('saved-state')
    saved_state.mkdir(exist_ok=True, parents=True)
    workflow_manager = WorkflowManager(
        JsonMetadataSerializer(saved_state),
        JsonEventSerializer(saved_state),
        {
            'DuckResponseFlow': DiscordWorkflowSerializer(DuckResponseFlow, client, saved_state)
        }
    )
    # give the workflow manager to the client
    client.set_workflow_manager(workflow_manager)
    # run the client
    with client:
        client.run(os.getenv('DISCORD_TOKEN'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=Path, default='prompts')
    parser.add_argument('--log-file', type=Path, default='/tmp/duck.log')
    args = parser.parse_args()
    main(args.prompts, args.log_file)
