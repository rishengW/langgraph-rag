"""Probe 2: env file with exact .env name - blocked on read/edit?"""
import os, sys, tempfile
from pathlib import Path

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

    # 1. Create a file named exactly .env
    created = create.invoke({
        'type': 'tool_call', 'name': create.name,
        'args': {'filename': '.env', 'content': 'FOO=bar\n'},
        'id': 'c1'
    })
    print('CREATE .env:', created.content[:150])
    print('file exists:', (session_root / '.env').exists())

    # 2. Try to read .env back
    r = read.invoke({'path': 'chat_uploads/t1/.env'})
    print('\nREAD .env:', r[:200])

    # 3. Inspect
    insp = inspect.invoke({'path': '.env'})
    print('\nINSPECT .env:', insp[:200])

    # 4. Also test with filename that ends in .env (blocked by name check?)
    print('\n--- Now create foo.env.local ---')
    c2 = create.invoke({
        'type': 'tool_call', 'name': create.name,
        'args': {'filename': 'x.env.local', 'content': 'A=1\n'},
        'id': 'c2'
    })
    print('CREATE x.env.local:', c2.content[:150])
    r2 = read.invoke({'path': 'chat_uploads/t1/x.env.local'})
    print('READ x.env.local:', r2[:200])

    # 5. Create x.env.example
    c3 = create.invoke({
        'type': 'tool_call', 'name': create.name,
        'args': {'filename': 'x.env.example', 'content': 'A=1\n'},
        'id': 'c3'
    })
    print('CREATE x.env.example:', c3.content[:150])
    r3 = read.invoke({'path': 'chat_uploads/t1/x.env.example'})
    print('READ x.env.example:', r3[:200])
