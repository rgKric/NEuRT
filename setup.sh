#!/bin/bash
python -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "venv was generated successfully"
echo "Start: python -m scripts.train_cls scripts/example_config_cls.json ..."
