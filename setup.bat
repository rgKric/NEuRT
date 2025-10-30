@echo off
REM venv generating
python -m venv venv

REM venv activating
call venv\Scripts\activate

REM libraries installing
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo venv was generated successfully
echo Start: python -m scripts.train_cls scripts/example_config_cls.json ...
pause