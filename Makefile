.PHONY: export-req

export-req:
	poetry export --without-hashes --without-urls | awk '{ print $$1 }' FS=';' > requirements.txt
