import re

def convert_cite(text: str) -> str:
    """
    Convert all occurrences of _cite[xxx] in the text to \cite{xxx}.
    
    Args:
        text: The input string containing _cite[xxx] patterns.
        
    Returns:
        A string with all _cite[xxx] replaced by \cite{xxx}.
    """
    # The pattern matches _cite followed by square brackets containing any characters except ]
    pattern = r'_cite\[(.*?)\]'
    replacement = r'\\cite{\1}'
    return re.sub(pattern, replacement, text)

print('\n', convert_cite(input('> ')))
