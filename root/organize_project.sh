#!/bin/bash

# Create standard project structure
echo "🔧 Creating folders..."
mkdir -p data notebooks src outputs

# Move the cleaned dataset (update the filename as needed)
echo "📁 Moving final cleaned dataset..."
mv data/hotel_no_show_cleaned.csv data/hotel_no_show_final_cleaned.csv 2>/dev/null

# Move EDA notebook to notebooks/
echo "📁 Moving EDA notebook..."
mv root/eda.ipynb notebooks/ 2>/dev/null

# Move src Python files
echo "📁 Moving source code scripts to src/..."
mv src/*.py src/ 2>/dev/null

# Move config.yaml if it exists
echo "📁 Moving config.yaml..."
mv root/config.yaml ./ 2>/dev/null

# Move README and requirements.txt if in root/
echo "📁 Moving README and requirements.txt..."
mv root/README.md ./ 2>/dev/null
mv root/requirements.txt ./ 2>/dev/null

# Clean up empty folders
echo "🧹 Removing empty folders..."
rmdir root 2>/dev/null

echo "✅ Project structure organized successfully!"
