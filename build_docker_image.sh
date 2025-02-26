# !/bin/sh

# Build docker image from the Dockerfile
docker.exe build -t stivendias/dmml .

# Upload docker image to DockerHub (requires a configured account)
docker.exe push stivendias/dmml:latest
