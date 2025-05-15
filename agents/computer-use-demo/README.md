# Anthropic Computer Use Demo

### Vertex

You'll need to pass in Google Cloud credentials with appropriate permissions to use Claude on Vertex.

```bash

gcloud auth application-default login
export VERTEX_REGION=""
export VERTEX_PROJECT_ID=""
./launch_with_split_prompts.sh

```

Once the container is running, see the [Accessing the demo app](#accessing-the-demo-app) section below for instructions on how to connect to the interface.

This example shows how to use the Google Cloud Application Default Credentials to authenticate with Vertex.
You can also set `GOOGLE_APPLICATION_CREDENTIALS` to use an arbitrary credential file, see the [Google Cloud Authentication documentation](https://cloud.google.com/docs/authentication/application-default-credentials#GAC) for more details.