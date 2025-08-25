venv_dir := ./.venv
requirements := requirements.txt
pip := $(venv_dir)/bin/pip
conda_dir := ./.conda

setup_conda:
	@mamba create --prefix $(conda_dir) python=3.10 --yes 

clean_conda:
	@rm -r $(conda_dir)

setup_venv:
	@python3 -m venv $(venv_dir)
	@$(pip) install --upgrade pip
	@$(pip) install -r $(requirements)

clean_venv:
	@rm -r $(venv_dir)
