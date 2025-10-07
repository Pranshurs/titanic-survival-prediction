#!/bin/bash
set -e

echo "🚀 Starting Titanic Survival pipeline..."
python -m titanic_survival.src.models.train
echo "✅ Pipeline completed successfully!"