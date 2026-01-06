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


## Badges 
We are unable to provide resources to run the artifacts due to resource constraints.
We are attempting to achieve the following badges:

- Artifacts Available
- Artifacts Functional