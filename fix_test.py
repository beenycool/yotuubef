import re

with open("tests/test_core_placeholders.py", "r") as f:
    content = f.read()

# Replace manager._db_manager = db_manager with manager._get_db_manager = lambda: db_manager
content = re.sub(
    r'manager\._db_manager = db_manager',
    r'manager._get_db_manager = lambda: db_manager',
    content
)

with open("tests/test_core_placeholders.py", "w") as f:
    f.write(content)
