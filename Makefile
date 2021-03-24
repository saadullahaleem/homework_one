up:
	docker-compose -f docker-compose.yaml up

upnew:
	docker-compose -f docker-compose.yaml build
	docker-compose -f docker-compose.yaml up
