#!/usr/bin/env python3
"""
TBot - AI-Powered Command Line Tool
A command-line tool that leverages AI models for various tasks like summarization, 
analysis, conversion, and more.
"""

import argparse
import sys
import logging
from typing import Optional
from pathlib import Path

# Import the HTTP client
try:
    from tbot.http_client import TBotClient
except ImportError:
    # Handle relative import if running as script
    sys.path.insert(0, str(Path(__file__).parent))
    from http_client import TBotClient

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Less verbose by default for CLI tool
    format='%(name)s: %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TBotCLI:
    """AI-powered command-line tool"""
    
    def __init__(self, 
                 conductor_url: str = "http://localhost:8001",
                 timeout: int = 60):
        self.client = TBotClient(
            conductor_url=conductor_url,
            timeout=timeout
        )
    
    def health_check(self) -> bool:
        """Check if servers are available"""
        try:
            health = self.client.health_check()
            conductor_status = health['conductor']['status']
            return conductor_status == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _ensure_healthy(self) -> None:
        """Ensure servers are healthy, exit if not"""
        if not self.health_check():
            print("❌ Error: TBot servers are not available.", file=sys.stderr)
            print("   Please start the servers with: poetry run tbot-servers", file=sys.stderr)
            sys.exit(1)
    
    def _process_with_ai(self, prompt: str, category: str = "general_reasoning") -> str:
        """Process a prompt with AI and return the response"""
        try:
            result = self.client.process_prompt(
                prompt=prompt,
                category=category,
                max_tokens=2048  # Reasonable limit for CLI output
            )
            
            if 'error' in result:
                raise RuntimeError(result['error'])
            
            return result.get('response', 'No response generated')
            
        except Exception as e:
            logger.error(f"AI processing failed: {e}")
            raise RuntimeError(f"AI processing failed: {str(e)}")
    
    def summarize(self, file_path: str, length: str = "medium") -> None:
        """Summarize a file"""
        self._ensure_healthy()
        
        try:
            # Read the file
            path = Path(file_path)
            if not path.exists():
                print(f"❌ Error: File '{file_path}' not found", file=sys.stderr)
                sys.exit(1)
            
            content = path.read_text(encoding='utf-8')
            
            # Prepare prompt based on length preference
            length_instructions = {
                "short": "in 2-3 sentences",
                "medium": "in 1-2 paragraphs", 
                "long": "in detail with key points"
            }
            
            length_inst = length_instructions.get(length, length_instructions["medium"])
            
            prompt = f"""Please summarize the following content {length_inst}:

{content}

Summary:"""
            
            print(f"📄 Summarizing {file_path}...")
            summary = self._process_with_ai(prompt, "general_reasoning")
            print(summary)
            
        except Exception as e:
            print(f"❌ Error summarizing file: {e}", file=sys.stderr)
            sys.exit(1)
    
    def convert_to_table(self, input_text: str) -> None:
        """Convert input text to a table format"""
        self._ensure_healthy()
        
        try:
            prompt = f"""Convert the following text into a well-formatted table. 
Use appropriate headers and organize the data logically:

{input_text}

Table:"""
            
            print("📊 Converting to table...")
            table = self._process_with_ai(prompt, "general_reasoning")
            print(table)
            
        except Exception as e:
            print(f"❌ Error converting to table: {e}", file=sys.stderr)
            sys.exit(1)
    
    def deduplicate_and_report(self, input_text: str) -> None:
        """Deduplicate errors and create a report"""
        self._ensure_healthy()
        
        try:
            prompt = f"""Analyze the following log/error data:
1. Identify and deduplicate similar errors/issues
2. Count occurrences of each unique error
3. Categorize by severity/type
4. Create a summary report

Data:
{input_text}

Report:"""
            
            print("🔍 Analyzing and deduplicating...")
            report = self._process_with_ai(prompt, "general_reasoning")
            print(report)
            
        except Exception as e:
            print(f"❌ Error creating report: {e}", file=sys.stderr)
            sys.exit(1)
    
    def analyze(self, input_text: str, analysis_type: str = "general") -> None:
        """Analyze text for patterns, insights, or specific analysis"""
        self._ensure_healthy()
        
        try:
            analysis_prompts = {
                "general": "Analyze the following text and provide insights, patterns, and key observations:",
                "sentiment": "Analyze the sentiment and emotional tone of the following text:",
                "structure": "Analyze the structure, organization, and format of the following text:",
                "keywords": "Extract and analyze the key terms, concepts, and important keywords from:",
                "security": "Analyze the following for potential security issues, vulnerabilities, or concerns:"
            }
            
            prompt_start = analysis_prompts.get(analysis_type, analysis_prompts["general"])
            
            prompt = f"""{prompt_start}

{input_text}

Analysis:"""
            
            print(f"� Performing {analysis_type} analysis...")
            analysis = self._process_with_ai(prompt, "general_reasoning")
            print(analysis)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}", file=sys.stderr)
            sys.exit(1)
    
    def explain(self, input_text: str, level: str = "simple") -> None:
        """Explain complex content in simpler terms"""
        self._ensure_healthy()
        
        try:
            level_instructions = {
                "simple": "in simple, easy-to-understand terms",
                "technical": "with technical details and context",
                "beginner": "as if explaining to a complete beginner"
            }
            
            level_inst = level_instructions.get(level, level_instructions["simple"])
            
            prompt = f"""Explain the following content {level_inst}:

{input_text}

Explanation:"""
            
            print(f"💡 Explaining {level}...")
            explanation = self._process_with_ai(prompt, "general_reasoning")
            print(explanation)
            
        except Exception as e:
            print(f"❌ Error explaining content: {e}", file=sys.stderr)
            sys.exit(1)
    
    def transform(self, input_text: str, target_format: str) -> None:
        """Transform text to different formats"""
        self._ensure_healthy()
        
        try:
            prompt = f"""Transform the following text to {target_format} format:

{input_text}

Transformed output:"""
            
            print(f"🔄 Transforming to {target_format}...")
            transformed = self._process_with_ai(prompt, "general_reasoning")
            print(transformed)
            
        except Exception as e:
            print(f"❌ Error transforming content: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    """Main entry point for TBot CLI tool"""
    parser = argparse.ArgumentParser(
        description="TBot - AI-Powered Command Line Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tbot summarize README.md
  tbot summarize --length short ARCHITECTURE.md
  cat errors.log | tbot convert-to-table
  cat errors.log | tbot deduplicate-errors
  tbot analyze --type security config.yaml
  tbot explain --level beginner complex_code.py
  tbot transform --format json data.csv
  
For more help on a specific command:
  tbot <command> --help
        """
    )
    
    # Global options
    parser.add_argument("--conductor-url", default="http://localhost:8001",
                       help="Conductor server URL (default: http://localhost:8001)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Request timeout in seconds (default: 60)")
    parser.add_argument("--health", action="store_true",
                       help="Check server health and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Summarize command
    summary_parser = subparsers.add_parser('summarize', help='Summarize a file')
    summary_parser.add_argument('file', help='File to summarize')
    summary_parser.add_argument('--length', choices=['short', 'medium', 'long'], 
                               default='medium', help='Summary length')
    
    # Convert to table command
    table_parser = subparsers.add_parser('convert-to-table', help='Convert input to table format')
    table_parser.add_argument('--input', help='Input text (if not using stdin)')
    
    # Deduplicate errors command  
    dedup_parser = subparsers.add_parser('deduplicate-errors', help='Deduplicate errors and create report')
    dedup_parser.add_argument('--input', help='Input text (if not using stdin)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze text for insights')
    analyze_parser.add_argument('--type', choices=['general', 'sentiment', 'structure', 'keywords', 'security'],
                               default='general', help='Type of analysis')
    analyze_parser.add_argument('--input', help='Input text (if not using stdin)')
    analyze_parser.add_argument('file', nargs='?', help='File to analyze')
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Explain complex content')
    explain_parser.add_argument('--level', choices=['simple', 'technical', 'beginner'],
                               default='simple', help='Explanation level')
    explain_parser.add_argument('--input', help='Input text (if not using stdin)')
    explain_parser.add_argument('file', nargs='?', help='File to explain')
    
    # Transform command
    transform_parser = subparsers.add_parser('transform', help='Transform text to different formats')
    transform_parser.add_argument('--format', required=True, 
                                 help='Target format (e.g., json, yaml, markdown, csv)')
    transform_parser.add_argument('--input', help='Input text (if not using stdin)')
    transform_parser.add_argument('file', nargs='?', help='File to transform')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Create TBot CLI instance
    tbot = TBotCLI(
        conductor_url=args.conductor_url,
        timeout=args.timeout
    )
    
    # Health check mode
    if args.health:
        print("🔍 Checking TBot server health...")
        healthy = tbot.health_check()
        if healthy:
            print("✅ TBot servers are healthy")
            sys.exit(0)
        else:
            print("❌ TBot servers are not available")
            print("   Start servers with: poetry run tbot-servers")
            sys.exit(1)
    
    # Handle commands
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'summarize':
            tbot.summarize(args.file, args.length)
        
        elif args.command == 'convert-to-table':
            input_text = _get_input_text(args.input)
            tbot.convert_to_table(input_text)
        
        elif args.command == 'deduplicate-errors':
            input_text = _get_input_text(args.input)
            tbot.deduplicate_and_report(input_text)
        
        elif args.command == 'analyze':
            input_text = _get_input_text(args.input, args.file)
            tbot.analyze(input_text, args.type)
        
        elif args.command == 'explain':
            input_text = _get_input_text(args.input, args.file)
            tbot.explain(input_text, args.level)
        
        elif args.command == 'transform':
            input_text = _get_input_text(args.input, args.file)
            tbot.transform(input_text, args.format)
        
        else:
            print(f"❌ Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled", file=sys.stderr)
        sys.exit(1)


def _get_input_text(input_arg: Optional[str] = None, file_arg: Optional[str] = None) -> str:
    """Get input text from argument, file, or stdin"""
    
    # If input is provided as argument
    if input_arg:
        return input_arg
    
    # If file is provided
    if file_arg:
        try:
            path = Path(file_arg)
            if not path.exists():
                print(f"❌ Error: File '{file_arg}' not found", file=sys.stderr)
                sys.exit(1)
            return path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"❌ Error reading file '{file_arg}': {e}", file=sys.stderr)
            sys.exit(1)
    
    # Try to read from stdin
    try:
        if sys.stdin.isatty():
            print("❌ Error: No input provided. Use --input, provide a file, or pipe data", file=sys.stderr)
            sys.exit(1)
        
        return sys.stdin.read()
    except KeyboardInterrupt:
        print("\n👋 Input cancelled", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
        sys.exit(0)
