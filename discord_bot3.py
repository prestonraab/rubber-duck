import json
import logging
import os
import signal
import subprocess
from pathlib import Path

import discord
import openai
import argparse

from typing import Callable, TypedDict, Union, List

from quest import event, signal as quest_signal
from quest.workflow_manager import WorkflowSerializer, WorkflowManager
from quest.json_seralizers import JsonMetadataSerializer, JsonEventSerializer
from quest.workflow import WorkflowFunction

logging.basicConfig(level=logging.DEBUG)
INPUT_EVENT_NAME = 'input'


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


class DuckResponseFlow:
    def __init__(self, thread: discord.Thread, author, chat_messages: list[GPTMessage],
                 control_channels: list[discord.TextChannel]):
        self.thread = thread
        self.author = author
        self.chat_messages = chat_messages
        self.control_channels = control_channels

    async def __call__(self, user_mention):
        welcome = f'{self.author.mention} What can I do for you?'

        async with self.thread.typing():
            await self.thread.send(welcome)

    @event
    async def display(self, text: str):
        await self.thread.send(text)

    @event
    async def display_control(self, text: str):
        for channel in self.control_channels:
            await channel.send(text)

    @event
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
            await send(self.thread, response)

    @quest_signal(INPUT_EVENT_NAME)
    async def query(self, message_text: str):
        """
        Query the OPENAI API
        """
        logging.debug(f"User said: {message_text}")

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

    @quest_signal(INPUT_EVENT_NAME)
    def get_input(self):
        ...


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


class DuckControlFlow:
    def __init__(self, message, control_channels: List[discord.TextChannel]):
        self.message = message
        self.control_channels = control_channels

    async def __call__(self):
        await self.control_on_message()

    @event
    async def control_on_message(self):
        """
        This function is called whenever the bot sees a message in a control channel
        """
        content = self.message.content
        if content.startswith('!restart'):
            await self.restart(self.message)

        elif content.startswith('!log'):
            await self.message.channel.send(file=discord.File('/tmp/duck.log'))

        elif content.startswith('!rmlog'):
            await self.execute_command("rm /tmp/duck.log", self.message.channel)
            await self.execute_command("touch /tmp/duck.log", self.message.channel)

        elif content.startswith('!status'):
            await self.message.channel.send('I am alive.')

        elif content.startswith('!help'):
            await self.display_help()
        elif content.startswith('!'):
            await self.message.channel.send('Unknown command. Try !help')

    @event
    async def display(self, text: str):
        for channel in self.control_channels:
            await send(channel, text)

    @event
    async def display_help(self):
        await self.display(
            "!restart - restart the bot\n"
            "!log - print the log file\n"
            "!rmlog - remove the log file\n"
            "!status - print a status message\n"
            "!help - print this message\n"
        )

    @event
    async def execute_command(self, text, channel):
        """
        Execute a command in the shell and return the output to the channel
        """
        # Run command using shell and pipe output to channel
        work_dir = Path(__file__).parent
        await self.display(f"```ps\n$ {text}```")
        process = subprocess.run(text, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir)
        # Get output of command and send to channel
        errors = process.stderr.decode('utf-8')
        if errors:
            await self.display(f'Errors: ```{errors}```')
        output = process.stdout.decode('utf-8')
        if output:
            await self.display(channel, f'```{output}```')
        return

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
        self._load_control_channels()
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
            await self._control_duck(message)
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

    # Start new DuckControlFlow
    # Maybe we don't need to start a workflow for this?
    async def _control_duck(self, message: discord.Message):
        await self.workflow_manager.start_async_workflow(str(message.channel.id),
                                                         DuckControlFlow(message, self.control_channels))

    async def _create_conversation(self, prefix, message: discord.Message):
        # Create a private thread in the message channel
        thread = await message.channel.create_thread(
            name=message.author.name,
            auto_archive_duration=60
        )

        messages = [
            dict(role='system', content=prefix or message.content)
        ]
        # start new register user workflow
        await self.workflow_manager.start_async_workflow(
            str(thread.id),
            DuckResponseFlow(thread, message.author, messages, self.control_channels),
            str(message.author.mention))
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
        {'DuckResponseFlow': DiscordWorkflowSerializer(DuckResponseFlow, client, saved_state)}
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
