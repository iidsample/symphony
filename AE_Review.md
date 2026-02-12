# Instructions of AE Review

## Resource Requirements

To run Symphony, in the paper we used 2 nodes with 4GPUs each. 

## Easiest way to run Symphony

Please download the provided Docker images and run the Symphony orchestrator script-**symphony-llama3-8b.py**
You should modify the ip addresses in the script to match your environment. 

## Code details

Major Changes
(i) vLLM: We have significant modified the vLLM code to support Symphony. We have made majority changes in the block manager and the scheduler. We have also added additional APIs to extract information and control vLLM behavior.
(ii) Node Manager:This is our interface to managing local memory and cache. It is responsible for evicting layers from cache and managing memory usage and integrating with vLLM.
(iii) Scheduler: This is our global scheduler. It is responsible for load-based scheduling and integrating with vLLM.


## Running single AE 
Based on the AE's request, we have created a single request GPU version of Symphony.
To run this single request version.

* docker pull codebedouin/symphony
* Start a terminal multiplexera like tmux/screen
* docker run --gpus all -it --rm --ipc=host --network=host --ulimit stack=67108864  codebedouin/symphony
* start two more screen 
* docker exec -it containerName bash
* Launch following scripts
* python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --gpu-memory-utilization 0.3

* python scheduler_queue_each_server.py
* python workloadgen.py

Please let us know if this works out. 
The above set of sequences run symphony on 100 requests. All the files are dependencies are pre-installed in the docker container.


## Badges 
We are unable to provide resources to run the artifacts due to resource constraints.
We are attempting to achieve the following badges:

- Artifacts Available
- Artifacts Functional
