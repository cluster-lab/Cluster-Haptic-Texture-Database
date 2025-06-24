xhost +local:docker
docker run -it \
           --rm -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $HOME/.Xauthority:/root/.Xauthority:rw \
           --gpus all \
           --shm-size=8g \
           -v $(pwd):/workspace \
            cluster_haptic_texture_database