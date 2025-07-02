prompt = 'where in the world is carmen san diego?'
prompt_lower = prompt.lower()
print('Prompt:', prompt)
print('Lowercase:', prompt_lower)

qa_keywords = ['what is', 'what are', 'where is', 'where are', 'when is', 'when was', 'who is', 'who was', 'how is', 'how does', 'why is', 'why does', 'explain', 'tell me about']
matches = [kw for kw in qa_keywords if kw in prompt_lower]
print('QA matches:', matches)

code_keywords = ['function', 'code', 'class', 'program', 'algorithm', 'script', 'def ', 'import ', 'const ', 'var ', 'write a']
code_matches = [kw for kw in code_keywords if kw in prompt_lower]
print('Code matches:', code_matches)

# Test the specific check
has_where_is = 'where is' in prompt_lower
print(f'Contains "where is": {has_where_is}')

# Test all QA patterns
for kw in qa_keywords:
    if kw in prompt_lower:
        print(f'Found keyword: "{kw}"')

