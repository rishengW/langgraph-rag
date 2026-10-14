"""Probe 1: env file create-then-edit round trip."""
import os, sys, tempfile, traceback
from pathlib import Path

os.environ.setdefault('DASHSCOPE_API_KEY', 'dummy')
sys.path.insert(0, os.path.abspath('../../..'))

from src.config.settings import Settings

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    file_root = td / 'files'
    session_root = file_root / 'chat_uploads' / 't1'
    session_root.mkdir(parents=True)
    s = Settings(dashscope_api_key='dummy', file_read_enabled=True,
                 file_read_root=str(file_root), env_edit_enabled=True)

    from src.backend.tools.env_edit import build_env_edit_tools
    from src.backend.tools.env_file import build_env_file_tool

    create, inspect, edit = build_env_edit_tools(s, session_root=session_root, thread_id='t1')
    read = build_env_file_tool(s)

    # 1. Create a .env file
    created = create.invoke({
        'type': 'tool_call', 'name': create.name,
        'args': {'filename': 'prod.env', 'content': 'FOO=bar\nDB_URL=postgres://x\n'},
        'id': 'c1'
    })
    print('CREATE result:', created.content[:200])
    print('created artifact:', created.artifact)
    print('file exists:', (session_root / 'prod.env').exists())

    # 2. Try to read it back via read_env_file
    r = read.invoke({'path': 'chat_uploads/t1/prod.env'})
    print('\nREAD result:', r[:200])

    # 3. Try to inspect
    insp = inspect.invoke({'path': 'prod.env'})
    print('\nINSPECT result:', insp[:200])

    # 4. Try to edit
    ed = edit.invoke({
        'type': 'tool_call', 'name': edit.name,
        'args': {'path': 'prod.env', 'operations': [
            {'action': 'set_value', 'key': 'FOO', 'value': 'baz', 'expected_value': 'bar'}
        ]},
        'id': 'e1'
    })
    print('\nEDIT result:', ed.content[:200])
    print('file still exists:', (session_root / 'prod.env').exists())

    # 5. Same but path outside session root
    r2 = read.invoke({'path': '../prod.env'})
    print('\nREAD ../prod.env:', r2[:200])
