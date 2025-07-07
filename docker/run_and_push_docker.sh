# Default values
script_name=$0
DOCKER_IMAGE_NAME='nvcr.io/nvidian/isaac-sdk/isaac_arena'
TAG='development'

SCRIPT_DIR="$(dirname "${BASH_SOURCE}")"

push=false

while getopts ":hptn:" OPTION; do
    case $OPTION in

        p)
            push=true
            ;;
        t)
            TAG=${OPTARG}
            ;;
        n)
            DOCKER_IMAGE_NAME=${OPTARG}
            ;;
        h | *)
            echo "Helper script to build $DOCKER_IMAGE_NAME (default)"
            echo "Usage:"
            echo "$script_name -h"
            echo "$script_name -p"
            echo "$script_name -t <tag>"
            echo "$script_name -n <docker name>"
            echo ""
            echo "  -p Push image to $DOCKER_IMAGE_NAME after building"
            echo "  -h help (this output)"
            echo "  -t <tag> (default is $TAG)"
            echo "  -n <docker name> (default is $DOCKER_IMAGE_NAME)"
            exit 0
            ;;
    esac
done

# Display the values being used
echo "Using Docker name: $DOCKER_IMAGE_NAME"
echo "Using tag: $TAG"

# Login to Docker registry
docker login nvcr.io

# Build the Docker image with the specified or default name and tag
docker build --pull -t ${DOCKER_IMAGE_NAME}:${TAG} --file $SCRIPT_DIR/Dockerfile.isaac_arena $SCRIPT_DIR/..

if $push ; then
    echo "Pushing to $DOCKER_IMAGE_NAME:${TAG}"
    docker push $DOCKER_IMAGE_NAME:${TAG}
fi

docker run -it -e "ACCEPT_EULA=Y" ${DOCKER_IMAGE_NAME}:${TAG}
