import re
import sys

with open('app/streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

prefix = 'st.markdown("""\n<div style=\''
suffix = '""", unsafe_allow_html=True)'

# Find the start and end of the markdown block
start_idx = content.find(prefix)
if start_idx == -1:
    print("Could not find start block")
    sys.exit(1)

# Find the end of the block
end_idx = content.find(suffix, start_idx)
if end_idx == -1:
    print("Could not find end block")
    sys.exit(1)

old_html = content[start_idx + len(prefix):end_idx]

# Remove leading whitespaces on EVERY line
new_html = '\n'.join([line.lstrip() for line in old_html.split('\n')])

new_content = content[:start_idx] + prefix + new_html + suffix + content[end_idx + len(suffix):]

with open('app/streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully unindented HTML block.")
