# TBot - AI-Powered Command Line Tool

TBot is a command-line tool that leverages AI models for various tasks like summarization, analysis, conversion, and more.

## Usage Examples

### File Summarization
```bash
# Summarize a README file
tbot summarize README.md

# Create a short summary
tbot summarize --length short ARCHITECTURE.md

# Create a detailed summary  
tbot summarize --length long documentation.txt
```

### Text Analysis
```bash
# Analyze a configuration file for security issues
tbot analyze --type security config.yaml

# Analyze sentiment of customer feedback
cat feedback.txt | tbot analyze --type sentiment

# Extract keywords from a document
tbot analyze --type keywords research_paper.md
```

### Data Conversion
```bash
# Convert log data to a table format
cat server.log | tbot convert-to-table

# Deduplicate errors and create a report
cat errors.log | tbot deduplicate-errors
```

### Content Transformation
```bash
# Transform CSV data to JSON
tbot transform --format json data.csv

# Convert configuration to YAML
cat config.conf | tbot transform --format yaml
```

### Content Explanation
```bash
# Explain complex code in simple terms
tbot explain --level simple complex_algorithm.py

# Get technical explanation of a system
tbot explain --level technical architecture.md
```

## Server Management

Start the AI servers:
```bash
poetry run tbot-servers
```

Check server health:
```bash
tbot --health
```

## Command Reference

- `summarize` - Summarize files or text
- `analyze` - Analyze text for insights, sentiment, security issues, etc.
- `convert-to-table` - Convert data to table format
- `deduplicate-errors` - Deduplicate and analyze error logs
- `explain` - Explain complex content in simpler terms
- `transform` - Transform text to different formats

Each command supports `--help` for detailed options.
