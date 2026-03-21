# Infer the repository URL from the git remote
REPO_URL := $(shell git config --get remote.origin.url)

CNAME =


publish:
	@cp assets/favicon.png web/favicon.png
	@echo "Publishing to $(REPO_URL)"
ifdef CNAME
	@ghp-import -n -p --cname $(CNAME) -f ./web
else
	@ghp-import -n -p -f ./web
endif
	@rm web/favicon.png

runweb:
	@cp assets/favicon.png web/favicon.png
	@echo "Serving at http://localhost:8000"
	@python -m http.server 8000 --directory web; rm -f web/favicon.png

export-models:
	@python giotto/utils/export_onnx.py
