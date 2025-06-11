#!/bin/bash
# Red Flags Profits - Data Analysis Runner
# Quick script to run the data analysis

echo "🔍 Red Flags Profits - Data Analysis Module"
echo "==========================================="

# Check if parquet file exists
if [ ! -f "data/all_billionaires.parquet" ]; then
    echo "❌ Error: Parquet file not found at data/all_billionaires.parquet"
    echo "💡 Make sure you have the data file in the correct location"
    exit 1
fi

echo "📂 Found parquet data file"

# Create output directory if it doesn't exist
mkdir -p output

echo "🚀 Running data analysis..."

# Run the basic analyzer
if python3 data_analyzer.py --input data/all_billionaires.parquet --output-dir output; then
    echo "✅ Basic analysis completed!"
    
    # Run enhanced analysis if the files exist
    if [ -f "sparkline_generator.py" ] && [ -f "enhanced_analyzer.py" ]; then
        echo "🚀 Running enhanced analysis..."
        if python3 enhanced_analyzer.py --input data/all_billionaires.parquet --output-dir output --summary; then
            echo "✅ Enhanced analysis completed!"
        else
            echo "⚠️  Enhanced analysis failed, but basic analysis succeeded"
        fi
    else
        echo "ℹ️  Enhanced analysis components not found, basic analysis only"
    fi
    
    echo ""
    echo "📊 Generated files in output/:"
    ls -la output/
    
    echo ""
    echo "🎉 Analysis complete! Files ready for website generation."
    
else
    echo "❌ Analysis failed!"
    exit 1
fi
