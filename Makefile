.PHONY: export-req

export-req:
	uv export --no-hashes --format requirements-txt > requirements.txt
