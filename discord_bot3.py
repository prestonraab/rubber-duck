import json
import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import discord
from discord import ChannelType
import openai
import argparse

from typing import Callable, TypedDict, Union, Any, Optional

from quest import event, signal as quest_signal
from quest.workflow_manager import WorkflowSerializer, WorkflowManager
from quest.json_seralizers import JsonMetadataSerializer, JsonEventSerializer
from quest.workflow import WorkflowFunction

INPUT_EVENT_NAME = 'get_input'


def load_env():
    with open('secrets.env') as f:
        for line in f:
            key, value = line.strip().split('=')
            os.environ[key] = value


load_env()
openai.api_key = os.environ['OPENAI_API_KEY']

AI_ENGINE = 'gpt-4'
FAST_AI_ENGINE = 'gpt-3.5-turbo-0613'


class GPTMessage(TypedDict):
    role: str
    content: str
    name: Optional[str]


@dataclass
class GPTParameters:
    type = "object"
    required: list[str] = field(default_factory=list)
    properties: dict[str, dict[str:Any]] = field(default_factory=dict)


class GPTFunction(TypedDict):
    # See this website for how to specify parameter types:
    # https://json-schema.org/understanding-json-schema/reference/object.html#properties
    name: str
    description: str
    parameters: GPTParameters


class DuckResponseFlow:
    thread: discord.Thread
    chat_messages: list[GPTMessage]
    message_id: int
    control_channels: list[discord.TextChannel]

    def __init__(self, thread, message_id, control_channels: list[discord.TextChannel],
                 chat_messages: list[GPTMessage] = None):
        self.thread = thread
        self.chat_messages: list[GPTMessage] = chat_messages
        self.message_id = message_id
        self.control_channels = control_channels

    async def __call__(self):
        user_response = await self.prompt("How can I help you?")

        while True:
            async with self.thread.typing():
                self.chat_messages.append(GPTMessage(role='user', content=user_response))
                await self.act_on_category()
                response = await self.query()
                if not response:
                    response = 'RubberDuck encountered an error.'

            user_response = await self.prompt(response)

    async def prompt(self, prompt: str):
        await self.display(prompt)
        return await self.get_input()

    @event
    async def display(self, text: str):
        async with self.thread.typing():
            await send(self.thread, text)

    @quest_signal(INPUT_EVENT_NAME)
    def get_input(self):
        ...

    @event
    async def query(self):
        """
        Query the OPENAI API
        """

        completion = await openai.ChatCompletion.acreate(
            messages=self.chat_messages,
            model=AI_ENGINE
        )
        logging.debug(f"Completion: {completion}")

        response_message = completion.choices[0]['message']
        response = response_message['content'].strip()
        logging.debug(f"Response: {response}")

        self.chat_messages.append(response_message)
        return response

    async def act_on_category(self):
        p = f'''If the above conversation is over, delete the conversation history.
        If answering the above question requires assignment-specific context, retrieve that assignment.
        If the above question relates to a specific Python topic, retrieve context from the appropriate guide entry.'''

        functions = [
            GPTFunction(
                name='end_conversation',
                description='Delete the conversation history',
                parameters=GPTParameters()),
            GPTFunction(
                name='get_assignment',
                description='Retrieve an assignment',
                parameters=GPTParameters(
                    properties={
                        "assignment_name": {"type": "string"}
                    }
                )
            ),
            GPTFunction(
                name='get_context',
                description='Retrieve context from the appropriate guide entry',
                parameters=GPTParameters(
                    properties={
                        "topic": {"type": "string"}
                    }
                )
            )]

        completion = await openai.ChatCompletion.acreate(
            messages=self.chat_messages[-1:] + [GPTMessage(role='system', content=p)],
            model=FAST_AI_ENGINE,
            functions=functions,
            function_call='auto'
        )
        logging.debug(f"Completion: {completion}")

        response_message = completion.choices[0]['message']

        # Step 2: check if GPT wanted to call a function
        if response_message.get("function_call"):
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "end_conversation": self.end_conversation,
                "get_assignment": self.get_assignment,
                "get_context": self.get_context
            }
            function_name = response_message["function_call"]["name"]
            try:
                function_to_call = available_functions[function_name]
                function_args = json.loads(response_message["function_call"]["arguments"])
                function_response = await function_to_call(**function_args)
            except Exception as e:
                logging.error(f"Error calling function: {e}")
                function_response = f"Error calling {function_name}."

            # Step 4: send the info on the function call and function response to GPT
            self.chat_messages.append(
                GPTMessage(
                    role="system",
                    name=function_name,
                    content=function_response
                )
            )  # extend conversation with function response

    @event
    async def display_control(self, text: str):
        for channel in self.control_channels:
            await channel.send(text)

    async def end_conversation(self):
        await self.display_control("Conversation ended.")
        await self.thread.delete()
        return "Conversation ended."

    async def get_assignment(self, assignment_name: str):
        await self.display_control(f"Retrieving assignment {assignment_name}.")
        return f"Assignment {assignment_name} retrieved."

    async def get_context(self, topic: str):
        await self.display_control(f"Retrieving context for {topic}.")
        return f"Context retrieved for {topic}."


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
            current_fence = "" if current_fence else line

    if block:
        yield block


async def send(thread: Union[discord.Thread, discord.TextChannel], text: str):
    for block in parse_blocks(text):
        await thread.send(block)


async def display_help(thread: Union[discord.Thread, discord.TextChannel]):
    await thread.send(
        "!restart - restart the bot\n"
        "!hard restart - restart the bot and clear the workflows\n"
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


async def hard_restart(message):
    """
    Restart the bot and clear the workflows
    :param message: The message that triggered the restart
    """
    await execute_command('rm -f saved-state/*.json', message.channel)
    await restart(message)


class DiscordWorkflowSerializer(WorkflowSerializer):
    def __init__(self, create_workflow: Callable[[Any], WorkflowFunction],
                 discord_client: discord.Client, folder: Path):
        self.create_workflow = create_workflow
        self.folder = folder
        self.discord_client = discord_client

    def serialize_workflow(self, workflow_id: str, workflow: WorkflowFunction):
        # Serialize workflow to "workflow" + workflow_id + ".json"
        # workflow is the workflow_object, attributes can be tested for existence with hasattr
        metadata = {"tid": workflow_id, "wid": workflow_id}
        if hasattr(workflow, 'message_id'):
            metadata["message_id"] = workflow.message_id

        if hasattr(workflow, 'control_channels'):
            metadata["control_channels"] = [str(channel.id) for channel in workflow.control_channels]

        if hasattr(workflow, 'chat_messages'):
            metadata["chat_messages"] = [message for message in workflow.chat_messages]

        file_to_save = "workflow" + workflow_id + ".json"
        with open(self.folder / file_to_save, 'w') as file:
            json.dump(metadata, file)

    def deserialize_workflow(self, workflow_id: str) -> WorkflowFunction:
        # Loads workflow from "workflow" + workflow_id + ".json"
        # Uses the create_workflow function to create a new workflow object
        file_to_load = "workflow" + workflow_id + ".json"
        with open(self.folder / file_to_load) as file:
            workflow_metadata = json.load(file)
            kwargs = {}
            if 'tid' in workflow_metadata:
                kwargs['thread'] = self.get_thread(workflow_metadata['tid'])
            if 'message_id' in workflow_metadata:
                kwargs['message_id'] = workflow_metadata['message_id']
            if 'control_channels' in workflow_metadata:
                kwargs['control_channels'] = [self.discord_client.get_channel(int(channel_id)) for channel_id in
                                              workflow_metadata['control_channels']]
            if 'chat_messages' in workflow_metadata:
                kwargs['chat_messages'] = [GPTMessage(**message_dict) for message_dict in
                                           workflow_metadata['chat_messages']]

            # Unpacks the args dictionary into keyword arguments
            # The create_workflow function is the constructor of the workflow object
            return self.create_workflow(**kwargs)

    def get_thread(self, tid) -> discord.Thread:
        thread = self.discord_client.get_channel(int(tid))
        return thread


class MyClient(discord.Client):
    def __init__(self, prompt_dir: Path, config: Path, log_file: Path):
        # adding intents module to prevent intents error in __init__ method in newer versions of Discord.py
        intents = discord.Intents.default()  # Select all the intents in your bot settings as it's easier
        intents.message_content = True
        super().__init__(intents=intents)

        self.workflow_manager = None
        self.log_file = log_file
        self.config = config
        self._load_control_channels(config)

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

        # contextualize members
        self.workflow_manager.load_workflows()
        await self.workflow_manager.resume_workflows()
        # print out information when the bot wakes up
        logging.info('Logged in as')
        logging.info(self.user.name)
        logging.info(self.user.id)
        logging.info('------')

        self._load_control_channels(self.config)
        for channel in self.control_channels:
            await send(channel, 'Duck online')

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

        for attachment in message.attachments:
            if attachment.filename.endswith('.txt'):
                message.content += '\n' + (await attachment.read()).decode('utf-8')

        if message.content.startswith('//'):
            return

        if message.channel.id in self.control_channel_ids:
            await self.control_on_message(message, self.log_file)
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
        # Create a public thread
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
            DuckResponseFlow(thread, message.id, self.control_channels, messages)
        )

    async def _continue_conversation(self, thread_id, text: str):
        # Get the conversation
        result = await self.workflow_manager.signal_async_workflow(thread_id, INPUT_EVENT_NAME, text)
        if result is not None:
            logging.debug(f'Conversation {thread_id} complete.')

    def set_workflow_manager(self, workflow_manager: WorkflowManager):
        self.workflow_manager = workflow_manager

    def _load_control_channels(self, config_file: Path):
        with open(config_file) as file:
            config = json.load(file)
        self.control_channel_ids = config['control_channels']
        self.control_channels = [c for c in self.get_all_channels() if c.id in self.control_channel_ids]

    async def control_on_message(self, message, log_file: Path):
        """
        This function is called whenever the bot sees a message in a control channel
        """
        content = message.content
        channel = message.channel
        if content.startswith('!restart'):
            await restart(message)

        elif content.startswith('!hard restart'):
            await hard_restart(message)

        elif content.startswith('!log'):
            await channel.send(file=discord.File(log_file))
            await channel.send(str(log_file.read_text()[-1000:]))

        elif content.startswith('!rm log'):
            await execute_command("rm " + str(log_file), channel)
            await execute_command("touch " + str(log_file), channel)

        elif content.startswith('!status'):
            await channel.send('🦆')

        elif content.startswith('!help'):
            await display_help(channel)

        elif content.startswith('!'):
            await channel.send('Unknown command. Try !help')


def main(prompts: Path, log_file: Path, config: Path, saved_state: Path):
    # create client
    client = MyClient(prompts, config, log_file)
    # init workflow manager
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
    parser.add_argument('--prompts', '-p', type=Path, default='prompts')
    parser.add_argument('--log_file', '-l', type=Path, default='/tmp/duck.log')
    parser.add_argument('--save_folder', '-s', type=Path, default='saved-state')
    parser.add_argument('--config', '-c', type=Path, default='config.json')
    args = parser.parse_args()
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    main(args.prompts, args.log_file, args.config, args.save_folder)
