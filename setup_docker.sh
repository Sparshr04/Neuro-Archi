#!/bin/bash
# setup_docker.sh — Helper script to build and launch the environment

echo "🛑  Stopping old containers..."
docker compose down

echo "🏗️  Building Docker image (ROS 2 Humble + ML deps)..."
docker compose build

echo "🚀  Starting container in background..."
docker compose up -d

echo "✅  Done! Attaching to container shell..."
echo "    (Type 'exit' to detach, container will stay running)"
echo ""
echo "    Inside container, verify GUI with:  xeyes"
echo "    Build workspace with:               colcon build"
echo ""

docker exec -it neuro_fusion_dev bash
