"""The HR helpdesk agent the Model Router series routes for.

It is the single agent from Agents Part 8 with the three extra rules added in Part 9, so every
number here is comparable with the cost table at the end of Agents Part 11.
"""
from langchain.agents import create_agent

from helpdesk.llm import get_model
from helpdesk.search import search_policy
from helpdesk.tools import get_holidays, get_leave_balance, get_payslip, raise_ticket

EMPLOYEE = "The person chatting with you is employee E1042. Today is 2026-11-07."

SINGLE_PROMPT = (f"You are an HR helpdesk assistant. {EMPLOYEE} Use tools for every fact. "
                 "Use search_policy for rules and the other tools for the employee's own data. "
                 "Cite policy sections in square brackets.")

RULES = (" Raise a ticket only when the employee asks for HR to follow up."
         " If asked about another employee's data, say you can't share it and do nothing else."
         " If the employee does not say which month, ask before calling get_payslip.")

SYSTEM_PROMPT = SINGLE_PROMPT + RULES

TOOLS = [search_policy, get_leave_balance, get_holidays, get_payslip, raise_ticket]


def build_agent(model=None, system_prompt: str = SYSTEM_PROMPT, tools=None):
    """Build the helpdesk agent. Pass a chat model, or leave it out to use MODEL from .env.
    `tools` lets the router wrap the tools (Part 5 blocks writes on cheap tiers)."""
    return create_agent(model or get_model(), name="helpdesk", tools=tools or TOOLS, system_prompt=system_prompt)
