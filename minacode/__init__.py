"""minacode: A small terminal coding agent written in Python.

The implementation lives in focused submodules (``base``, ``session``,
``skill``, ``mcp``, ``tools``, ``engine``, ``tui``) plus a ``__main__`` entry
point.  The public names are
re-exported here so ``import minacode`` keeps exposing the same namespace the
single-file module used to provide.
"""

import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time

import code_symbol_index as csi
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys
from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from minacode import engine, tools, tui
from minacode.base import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    CHAT_REASONING_CHOICES,
    CHAT_REASONING_EFFORT_VALUES,
    DEEPSEEK_DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    DISMISSED,
    HTTP_USER_AGENT,
    Json,
    MAX_TOOL_OUTPUT_TOKENS,
    MIN_CONTEXT_SAFETY_TOKENS,
    MODEL_REQUEST_RETRIES,
    PROVIDER_API_CHOICES,
    REASONING_CHOICES,
    REASONING_LEVELS,
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
    Config,
    ConfigError,
    ConfigFile,
    MinacodeError,
    ModelError,
    ModelRequestRetry,
    ModelUsage,
    ProviderConfig,
    RuntimeSettings,
    SystemInfo,
    Text,
    ToolCall,
    ToolError,
    UpdateStatus,
    __version__,
)
from minacode.engine import (
    ActiveResource,
    Agent,
    ContextManager,
    EditBatchPlan,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    ModelClient,
    PreparedRequest,
    ToolDisplay,
    ToolRunner,
    TurnBox,
    UpdateChecker,
)
from minacode.mcp import MCPFileTokenStore, MCPManager, MCPResourceInfo, MCPServerConfig, MCPToolInfo
from minacode.session import (
    AgentState,
    BackgroundJob,
    HistorySegment,
    PlanItem,
    QueuedInput,
    Session,
    SessionSnapshotCodec,
    SessionSnapshotStore,
    ToolErrorRecord,
    ToolResultRecord,
    TurnDiff,
)
from minacode.skill import Skill, SkillLibrary
from minacode.tools import (
    TOOLS,
    TOOL_REGISTRY,
    AskSpec,
    AskTool,
    BashTool,
    CodeIndex,
    Edit,
    EditApplyResult,
    EditTool,
    InspectCodeTool,
    JobTool,
    MCPTool,
    NoteTool,
    ReadTool,
    RecallContextTool,
    RecallTool,
    SearchTool,
    SkillTool,
    Tool,
)
from minacode.tui import (
    TUI_MODAL_PENDING,
    BashLivePreview,
    CallbackPlaceholder,
    ChoiceViewState,
    CommandCompleter,
    CommandLoop,
    DiffViewState,
    StatusBar,
    TabbedViewState,
    Theme,
    TuiApp,
    TuiModal,
    TuiRuntime,
    UiPrinter,
)


def __getattr__(name: str):
    # Lazily expose the entry point so importing minacode (and running `python -m minacode`)
    # does not eagerly import __main__, which would raise a duplicate-module RuntimeWarning.
    if name == "main":
        from minacode.__main__ import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
