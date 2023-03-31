#!/bin/bash

# ....Run norlab-ros2-humble-develop:arm64-l4t-r35.2.1 for norlab-MPPI..............................................................
docker compose -f /home/snow/NorLab_MPPI/docker-compose.ros2.jetson.run.yaml up --detach --wait

if [[ -n $TEAMCITY_VERSION ]]; then
  # (NICE TO HAVE) ToDo: implement >> fetch container name from an .env file
  echo -e "${DS_MSG_EMPH_FORMAT}The container is running inside a TeamCity agent >> keep container detached${DS_MSG_END_FORMAT}"
else
  docker compose -f /home/snow/NorLab_MPPI/docker-compose.ros2.jetson.run.yaml exec develop /ros2_entrypoint.bash bash
fi
