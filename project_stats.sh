#!/bin/bash

echo "📊 TBot Project Statistics"
echo "========================="
echo ""

# Function to count files and lines
count_files_and_lines() {
    local path="$1"
    local pattern="$2"
    local files=$(find "$path" -name "$pattern" -not -path "./.venv/*" -not -path "./__pycache__/*" 2>/dev/null | wc -l)
    local lines=$(find "$path" -name "$pattern" -not -path "./.venv/*" -not -path "./__pycache__/*" -exec cat {} \; 2>/dev/null | wc -l)
    echo "$files files, $lines lines"
}

# Get component statistics
echo "🐍 Python Files Breakdown:"
echo "-------------------------"

conductor_files=$(find ./conductor -name "*.py" 2>/dev/null | wc -l)
conductor_lines=$(find ./conductor -name "*.py" -exec cat {} \; 2>/dev/null | wc -l)
echo "Conductor (LLM Server):     $conductor_files files, $conductor_lines lines"

# Conductor sub-components
engines_files=$(find ./conductor/engines -name "*.py" 2>/dev/null | wc -l)
engines_lines=$(find ./conductor/engines -name "*.py" -exec cat {} \; 2>/dev/null | wc -l)
echo "  - AI Engines:             $engines_files files, $engines_lines lines"

models_files=$(find ./conductor/model_info -name "*.py" 2>/dev/null | wc -l)
models_lines=$(find ./conductor/model_info -name "*.py" -exec cat {} \; 2>/dev/null | wc -l)
echo "  - Model Configurations:   $models_files files, $models_lines lines"

core_files=$(find ./conductor -name "*.py" -not -path "./conductor/engines/*" -not -path "./conductor/model_info/*" -not -path "./conductor/tests/*" 2>/dev/null | wc -l)
core_lines=$(find ./conductor -name "*.py" -not -path "./conductor/engines/*" -not -path "./conductor/model_info/*" -not -path "./conductor/tests/*" -exec cat {} \; 2>/dev/null | wc -l)
echo "  - Core Logic:             $core_files files, $core_lines lines"

diffusiond_files=$(find ./diffusiond -name "*.py" 2>/dev/null | wc -l)
diffusiond_lines=$(find ./diffusiond -name "*.py" -exec cat {} \; 2>/dev/null | wc -l)
echo "Diffusiond (Image Server):  $diffusiond_files files, $diffusiond_lines lines"

tbot_files=$(find ./tbot -name "*.py" 2>/dev/null | wc -l)
tbot_lines=$(find ./tbot -name "*.py" -exec cat {} \; 2>/dev/null | wc -l)
echo "TBot CLI Tool:              $tbot_files files, $tbot_lines lines"

test_files=$(find . -name "test_*.py" -not -path "./.venv/*" 2>/dev/null | wc -l)
test_lines=$(find . -name "test_*.py" -not -path "./.venv/*" -exec cat {} \; 2>/dev/null | wc -l)
echo "Tests:                      $test_files files, $test_lines lines"

root_files=$(find . -maxdepth 1 -name "*.py" 2>/dev/null | wc -l)
root_lines=$(find . -maxdepth 1 -name "*.py" -exec cat {} \; 2>/dev/null | wc -l)
echo "Root Level:                 $root_files files, $root_lines lines"

echo ""

# Total Python statistics
total_py_files=$(find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" 2>/dev/null | wc -l)
total_py_lines=$(find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" -exec cat {} \; 2>/dev/null | wc -l)
echo "🐍 Total Python:            $total_py_files files, $total_py_lines lines"

echo ""
echo "📝 Documentation:"
echo "----------------"

md_files=$(find . -name "*.md" 2>/dev/null | wc -l)
md_lines=$(find . -name "*.md" -exec cat {} \; 2>/dev/null | wc -l)
echo "Markdown Files:             $md_files files, $md_lines lines"

echo ""
echo "📊 Project Totals:"
echo "----------------"

total_files=$((total_py_files + md_files))
total_lines=$((total_py_lines + md_lines))

echo "Total Files:                $total_files files"
echo "Total Lines:                $total_lines lines"
echo "Code/Docs Ratio:            $(echo "scale=1; $total_py_lines * 100 / $total_lines" | bc -l)% code, $(echo "scale=1; $md_lines * 100 / $total_lines" | bc -l)% docs"

echo ""
echo "🎯 Project Composition:"
echo "----------------------"
echo "- AI-powered command-line tool with client-server architecture"
echo "- $engines_files specialized AI engines for different tasks"
echo "- $models_files model configurations supporting various LLMs"
echo "- Complete HTTP API with health monitoring and error handling"
echo "- Comprehensive documentation and test coverage"
echo ""
echo "Generated on: $(date)"
