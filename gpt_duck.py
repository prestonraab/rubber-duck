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

from typing import Callable, TypedDict, Union, Any, Awaitable

INPUT_EVENT_NAME = 'get_input'


def load_env():
    with open('secrets.env') as f:
        for line in f:
            key, value = line.strip().split('=')
            os.environ[key] = value


load_env()
openai.api_key = os.environ['OPENAI_API_KEY']

AI_ENGINE = 'gpt-3.5-turbo'  # 'gpt-4'
FAST_AI_ENGINE = 'gpt-3.5-turbo-0613'


class GPTMessage(TypedDict, total=False):
    role: str
    content: str
    name: str


def gpt_parameters(properties: dict[str, Any] = None, required: list[str] = None):
    d = {"type": "object"}
    if required:
        d["required"] = required
    d["properties"] = properties or {}
    return d


class GPTFunction(TypedDict):
    # See this website for how to specify parameter types:
    # https://json-schema.org/understanding-json-schema/reference/object.html#properties
    name: str
    description: str
    parameters: dict[str, dict[str, Any]]


class Duck:
    get_input: Callable[[], Awaitable[str]]
    send: Callable[[str], Awaitable[None]]
    log: Callable[[str], Awaitable[None]]
    chat_messages: list[GPTMessage]

    def __init__(self,
                 send: Callable[[str], Awaitable[None]],
                 log: Callable[[str], Awaitable[None]],
                 get_input: Callable[[], Awaitable[str]],
                 chat_messages: list[GPTMessage] = None):
        self.chat_messages = chat_messages
        self.send = send
        self.log = log
        self.get_input = get_input

    async def welcome(self):
        return await self.prompt("How can I help you?")

    async def add_user_message(self, message: str):
        self.chat_messages.append(GPTMessage(role='user', content=message))

    async def prompt(self, prompt: str):
        await self.send(prompt)
        return await self.get_input()

    async def query(self):
        """
        Query the OPENAI API
        """
        completion = await openai.ChatCompletion.acreate(
            messages=self.chat_messages,
            model=AI_ENGINE
        )

        response_message = completion.choices[0]['message']
        response = response_message['content'].strip()

        self.chat_messages.append(response_message)
        return response

    done_examples = [
        "Thanks!",
        "later",
        "another time"
        "I think I understand now.",
        "I'm good for now.",
        "That's it.",
        "Bye",
        "nevermind"
    ]
    example_assignments = [
        "homework 1a",
        "lab 2e",
        "project 3"
    ]
    example_problems = [
        "I need help with scurry",
        "I'm working on personal library",
        "I'm stuck on practice problems",
        "postal routing is hard",
        "Can you help me with auto correct",
        "I'm doing scheduling",
        "Help with simplify",
        "Help on moose",
        "Can you help me with invert-careful?"
    ]
    example_topics = [
        "functions",
        "while loops",
        "list patterns",
        "split and join",
        "grouping",
        "program arguments"
    ]

    categorize_prompt = f'''If the following conversation is over, delete the conversation history.
        Examples of last messages that indicate a conversation is over: {'“'}{'”, “'.join(done_examples)}{'”'}.
        If answering a question below requires assignment-specific context, retrieve that assignment.
        Examples of assignment names: {'“'}{'”, “'.join(example_assignments)}{'”'}.
        If a question below needs problem-specific context, and the assignment is unknown, retrieve that problem. A problem is part of an assignment.
        Examples of questions that would require retrieving a problem: {'“'}{'”, “'.join(example_problems)}{'”'}.
        If a question below relates to a specific Python topic, retrieve context from the appropriate guide entry.
        Examples of topics: {'“'}{'”, “'.join(example_topics)}{'”'}.'''

    async def act_on_category(self):
        functions = [
            GPTFunction(
                name='end_conversation',
                description='Delete the conversation history',
                parameters=gpt_parameters()
            ),
            GPTFunction(
                name='retrieve_assignment',
                description='Retrieve a specific assignment, formatted like “homework 1a”',
                parameters=gpt_parameters(
                    properties={
                        "assignment_name": {"type": "string"}
                    }
                )
            ),
            GPTFunction(
                name='retrieve_problem',
                description='Retrieve a problem from an unknown assignment',
                parameters=gpt_parameters(
                    properties={
                        "problem_name": {"type": "string"}
                    }
                )
            ),
            GPTFunction(
                name='retrieve_topic',
                description='Retrieve context from the appropriate guide entry',
                parameters=gpt_parameters(
                    properties={
                        "topic": {"type": "string"}
                    }
                )
            )]

        completion = await openai.ChatCompletion.acreate(
            # Prompt first, then last message
            messages=[GPTMessage(role='system', content=Duck.categorize_prompt)] + self.chat_messages[-1:],
            model=FAST_AI_ENGINE,
            functions=functions,
            function_call='auto'
        )

        response_message = completion.choices[0]['message']

        # Check if GPT wanted to call a function
        if response_message.get("function_call"):
            # Call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "end_conversation": self.end_conversation,
                "retrieve_assignment": self.get_assignment,
                "retrieve_problem": self.get_problem,
                "retrieve_topic": self.get_topic
            }
            function_name = response_message["function_call"]["name"]
            try:
                function_to_call = available_functions[function_name]
                function_args = json.loads(response_message["function_call"]["arguments"])
                function_response = await function_to_call(**function_args)
            except Exception as e:
                logging.error(f"Error calling function: {e}")
                function_response = f"Error calling {function_name}. \n{e}"

            # Add a system message to let GPT know that the function has been called
            self.chat_messages.append(
                GPTMessage(
                    role="system",
                    name=function_name,
                    content=function_response
                )
            )

    async def end_conversation(self):
        await self.log("Conversation ended.")
        await self.thread.delete()
        return "Conversation ended."

    async def get_assignment(self, assignment_name: str):
        await self.log(f"Retrieving assignment {assignment_name}.")
        return f"Assignment {assignment_name} retrieved."

    async def get_problem(self, problem_name: str):
        await self.log(f"Retrieving problem {problem_name}.")
        assignment_name = "homework 2a"
        await self.get_assignment(assignment_name)
        return f"Problem {problem_name} retrieved from assignment {assignment_name}."

    async def get_topic(self, topic: str):
        await self.log(f"Retrieving context for {topic}.")
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
