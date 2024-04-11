#!/usr/bin/env python3
#
# Authors:
#    wadih@wadih.com
#    pathfinder@wadih.com
#
# symsearch.py

import argparse
import os
import re
import json
import datetime
import subprocess
from subprocess import check_output, CalledProcessError
import pygments
from pygments.styles import get_all_styles
from pygments import highlight
from pygments.style import Style
from pygments.token import Token
from pygments.lexers import Python3Lexer, guess_lexer_for_filename, get_lexer_by_name, guess_lexer, get_all_lexers
from pygments.formatters import Terminal256Formatter

# Optional NLP imports with lazy loading
NLP_AVAILABLE = False
nlp, sia = None, None

def load_nlp_models():
    global nlp, sia, NLP_AVAILABLE
    try:
        import spacy
        from nltk.sentiment import SentimentIntensityAnalyzer
        nlp = spacy.load("en_core_web_sm")
        sia = SentimentIntensityAnalyzer()
        NLP_AVAILABLE = True
    except ImportError:
        check_nlp_packages()
        load_nlp_models()
        NLP_AVAILABLE = True
    except Exception as e:
        print(e)
        NLP_AVAILABLE = False

    return NLP_AVAILABLE

def check_nlp_packages():
    try:
        subprocess.call(['python3', '-m', 'spacy', 'download', 'en_core_web_sm'])
    except Exception as e:
        print(f"Error installing spacy en_core_web_sm: {e}")

    try:
        subprocess.call(['python3', '-m', 'nltk.downloader', 'vader_lexicon'])
    except Exception as e:
        print(f"Error installing nltk vader_lexicon: {e}")

    NLP_AVAILABLE = True


# ANSI escape codes for highlighting, used if not in JSONL mode
YELLOW_START = "\033[93m"
YELLOW_END = "\033[0m"

class CodeBlockIdentifier:
    def __init__(self, syntax_style='monokai'):
        self.syntax_style = syntax_style
        self.block_match = re.compile(
            r'```(?:\w*\n)?(.*?)```|~~~(?:\w*\n)?(.*?)~~~|\'\'\'(?:\w*\n)?(.*?)\'\'\'', re.DOTALL
        )

    def extract_code_blocks(self, text):
        code_blocks = [block for block in self.block_match.findall(text) if block]
        return [''.join(block).strip() for block in code_blocks]

        # Return the linter output
        #return process.stdout.decode()
        return code

    def highlight_code(self, code_block, language='auto', jsonl=False):
        try:
            lexer = guess_lexer(code_block) if language == 'auto' else get_lexer_by_name(language)
            formatter = Terminal256Formatter(style=self.syntax_style) if not jsonl else Terminal256Formatter(style="bw", nowrap=True)
        except Exception:
            lexer = get_lexer_by_name('text')
            formatter = Terminal256Formatter(style="bw", nowrap=True)

        return highlight(code_block, lexer, formatter)

def print_file_metadata(file_path):
    try:
        file_stat = os.stat(file_path)
        print(f"--- {file_path}")
        print(f"Size: {file_stat.st_size} bytes")
        print(f"Last modified: {datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat()}")
        print(f"Last accessed: {datetime.datetime.fromtimestamp(file_stat.st_atime).isoformat()}")
        print(f"Creation time: {datetime.datetime.fromtimestamp(file_stat.st_ctime).isoformat()}")
        print(f"Mode: {file_stat.st_mode}")
        print(f"Inode: {file_stat.st_ino}")
        print(f"Device: {file_stat.st_dev}")
        print(f"Number of links: {file_stat.st_nlink}")
        print(f"UID of owner: {file_stat.st_uid}")
        print(f"GID of owner: {file_stat.st_gid}")
    except Exception as e:
        print(f"Error obtaining metadata for {file_path}: {e}")
    except CalledProcessError as e:
        print(f"Error obtaining metadata for {file_path}: {e}")

def get_file_metadata(file_path):
    try:
        file_stat = os.stat(file_path)
        return {
            "size": file_stat.st_size,
            "last_modified": datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        }
    except OSError as e:
        return {"error": str(e)}

def highlight_text(text, term, jsonl=False):
    if jsonl:
        return text  # Do not apply ANSI colors in JSONL mode
    term_escaped = re.escape(term)
    return re.sub(f"({term_escaped})", YELLOW_START + r"\1" + YELLOW_END, text, flags=re.IGNORECASE)

def analyze_text_nlp(text):
    if NLP_AVAILABLE:
        sentiment = sia.polarity_scores(text)
        doc = nlp(text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        return {"sentiment": sentiment['compound'], "entities": entities}
    else:
        return {"error": "NLP analysis skipped. NLTK or spaCy not available."}

def search_directory(directory, query, before_context, after_context, use_nlp, extract_code, syntax_style, jsonl):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            search_file(file_path, query, before_context, after_context, use_nlp, extract_code, syntax_style, jsonl)

def search_file(file_path, query, before_context, after_context, use_nlp, extract_code, syntax_style, jsonl):
    code = False
    matches = []
    code_identifier = CodeBlockIdentifier(syntax_style=syntax_style)

    # Identify code files for code extraction.
    code_file_pattern = r'\.(py|html|c|cpp|js|java|cs|php|rb|go|tsx|ts|swift|kt|scala)$'

    # Check the file_path against the pattern
    if re.search(code_file_pattern, file_path):
        code = True
    else:
        code = False

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        
        if extract_code:
            if code is True:
                #matches.append({"type": "code", "content": text.strip()}) 
                highlighted = code_identifier.highlight_code(text, jsonl=jsonl)
                matches.append({"type": "code", "content": highlighted.strip()}) 
            else:
                code_blocks = code_identifier.extract_code_blocks(text)
                for block in code_blocks:
                    if re.search(query, block, re.IGNORECASE):
                        #matches.append({"type": "code", "content": block.strip()})
                        highlighted = code_identifier.highlight_code(block, jsonl=jsonl)
                        matches.append({"type": "code", "content": highlighted.strip()})
        else:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if re.search(query, line, re.IGNORECASE):
                    match_content = highlight_text(line.strip(), query, jsonl)
                    matches.append({"type": "text", "content": match_content})
                    if use_nlp and NLP_AVAILABLE:
                        nlp_analysis = analyze_text_nlp(line)
                        matches[-1].update(nlp_analysis)
                        
        if matches and jsonl:
            file_metadata = get_file_metadata(file_path)
            json_output = {"file": file_path, "metadata": file_metadata, "matches": matches}
            print(json.dumps(json_output))
        elif matches and not jsonl:
            print_file_metadata(file_path)
            for match in matches:
                print(match["content"])
                
    except Exception as e:
        if jsonl:
            print(json.dumps({"file": file_path, "error": str(e)}))
        else:
            print(f"Error reading file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Search files with optional code extraction and NLP analysis.")
    parser.add_argument("directory", help="Directory to search")
    parser.add_argument("query", help="Search query, supports basic regex")
    parser.add_argument("-B", "--before-context", type=int, default=2, help="Lines of context before a match")
    parser.add_argument("-A", "--after-context", type=int, default=2, help="Lines of context after a match")
    parser.add_argument("--code", action="store_true", help="Extract and highlight code blocks matching the query")
    parser.add_argument("--use-nlp", action="store_true", help="Perform NLP analysis on matched text, requires NLTK and spaCy")

    pygments_list = list(pygments.styles.get_all_styles())
    pygments_list.append('none')
    parser.add_argument("--style", default="monokai", choices=pygments_list, help="Syntax highlighting style for code")
    parser.add_argument("--jsonl", action="store_true", help="Output results in JSON Lines format for programmatic use")

    args = parser.parse_args()
    
    if args.use_nlp:
        NLP_AVAILABLE = load_nlp_models()
    else:
        NLP_AVAILABLE = False
    
    search_directory(args.directory, args.query, args.before_context, args.after_context, NLP_AVAILABLE, args.code, args.style, args.jsonl)

if __name__ == "__main__":
    main()
