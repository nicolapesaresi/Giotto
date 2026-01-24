# Infer the repository URL from the git remote
REPO_URL := $(shell git config --get remote.origin.url)

APP_NAME = 'Giotto'
ICON = 'giotto/assets/favicon.png'

CNAME = 


publish:
	@pygbag --build --app_name $(APP_NAME) --icon $(ICON) .
	@echo "Publishing to $(REPO_URL)"
ifdef CNAME
	@ghp-import -n -p --cname $(CNAME) -f ./build/web 
else
	@ghp-import -n -p -f ./build/web 
endif

runweb:
	@pygbag .