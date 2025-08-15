venv_dir := ./.venv
requirements := requirements.txt
pip := $(venv_dir)/bin/pip

setup:
	@python3 -m venv $(venv_dir)
	@$(pip) install --upgrade pip
	@$(pip) install -r $(requirements)

clean_setup:
	@rm -r $(venv_dir)
