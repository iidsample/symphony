# Symphony
Symphony contains multiple components for running turn-by-turn workloads like chatbots and agents.

## Components

- **Workload Generator**: A Python-based workload generator designed to simulate realistic multi-turn conversation workloads for LLM inference systems. This tool generates client sessions and sends requests to a scheduler via gRPC, modeling human-like typing and reading speeds.
- **Scheduler**: A global gRPC server that receives requests from the workload generator and forwards them to the LLM inference system.
- **Modified vLLM**: We have modified vLLM to support turn-by-turn workloads. It is added as a submodule

- **Node Scheduler**: Each instance of serving engine has a node level scheduler that receives requests from the scheduler and forwards them to the LLM inference system. It is responsible for managing the requests and memory on a single node.


## Workload Generator

A Python-based workload generator designed to simulate realistic multi-turn conversation workloads for LLM inference systems. This tool generates client sessions and sends requests to a scheduler via gRPC, modeling human-like typing and reading speeds.

### Overview

The workload generator simulates multiple concurrent clients engaging in multi-turn conversations with an LLM inference system. It models realistic user behavior by incorporating:

- **Typing speed**: How fast users type their prompts (default: 6 characters/second)
- **Reading speed**: How fast users read responses before typing the next query (default: 25 characters/second)
- **Session management**: Maintains a specified number of active concurrent sessions
- **Multi-turn conversations**: Handles conversational context across multiple request-response pairs

### Prerequisites

- Python 3.7+
- gRPC Python packages
- Dataset file with conversation data (JSON format)
- Access to the scheduler service via gRPC

### Installation

1. Install required dependencies:
```bash
pip install grpcio grpcio-tools
```

2. Ensure the gRPC stubs are generated and available in the `rpc/grpc_stubs/` directory


### Dataset Format

The workload generator expects a JSON dataset file containing conversation data. The dataset should be a list of conversation objects with the following structure:

```json
[
  {
    "conversations": [
      {"value": "First user prompt"},
      {"value": "Assistant response to first prompt"},
      {"value": "Second user prompt"},
      {"value": "Assistant response to second prompt"},
      ...
    ]
  },
  ...
]
```

**Important Notes:**
- Each conversation must have at least 2 turns (user prompt + response pairs)
- Conversations with only 1 turn are automatically skipped
- The dataset path is currently hardcoded in the code at line 33-35

### Usage

#### Basic Usage

Run the workload generator with default settings:

```bash
python workloadgen.py
```

#### Command-Line Arguments

```bash
python workloadgen.py \
  --reciever-port 50050 \
  --scheduler-ip localhost \
  --scheduler-port 50051 \
  --num-clients 256
```

##### Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--reciever-port` | str | 50050 | Port for the workload generator's response receiver server |
| `--scheduler-ip` | str | localhost | IP address of the scheduler service |
| `--scheduler-port` | str | 50051 | Port of the scheduler service |
| `--num-clients` | int | 256 | Number of concurrent client sessions to maintain |

#### Example Scenarios

**Single client for testing:**
```bash
python workloadgen.py --num-clients 1
```

**High load testing with 1000 concurrent clients:**
```bash
python workloadgen.py --num-clients 1000
```

**Connect to remote scheduler:**
```bash
python workloadgen.py \
  --scheduler-ip 192.168.1.100 \
  --scheduler-port 8080 \
  --num-clients 128
```

### How It Works

#### Workflow

1. **Initialization**:
   - Launches a gRPC server to receive responses from the scheduler
   - Establishes a gRPC client connection to the scheduler
   - Loads the conversation dataset from the JSON file

2. **Session Management**:
   - Maintains exactly `num_clients` active sessions at all time
   - When a session completes, a new session is started from the dataset
   - Each session is assigned a unique session ID

3. **Request Timing**:
   - For **first requests**: Only typing time is considered (reading time = 0)
   - For **subsequent requests**: 
     - Reading time = length of previous response / read_speed_char_ps
     - Typing time = length of current prompt / type_speed_char_ps
     - Total wait time = reading time + typing time

4. **Request Scheduling**:
   - Calculates wait times for all pending requests
   - Sleeps until the next request is ready
   - Sends the request with the minimum wait time
   - Updates all other wait times accordingly

5. **Response Handling**:
   - Receives responses via the gRPC server
   - Prepares the next request in the conversation
   - Marks sessions as complete when the last request is sent

#### Key Components

- **`WorkloadGen` class**: Main workload generator orchestrating the entire process
- **`send_data()` method**: Core loop managing session lifecycle and request sending
- **`update_time_to_next_req()` method**: Calculates timing based on reading/typing speeds
- **`launch_server()` function**: Initializes the response receiver server

### Configuration

#### Adjusting User Behavior

You can modify the simulated user behavior by changing these constants in the `__init__` method (lines 58-61):

```python
self.read_speed_char_ps = 25  # Characters read per second
self.type_speed_char_ps = 6   # Characters typed per second
```

#### Dataset Path

Update the dataset path at lines 33-35:

```python
self.dataset_path = "/path/to/your/dataset.json"
```

### Output

The workload generator provides console output showing:

- Number of active sessions
- Number of new clients needed
- Received responses
- Time to next request for each session

Example output:
```
Server Started
Length of active_sessions 0
New clients needed 256
Returned Response []
Time to next rq {0: 15.2, 1: 18.5, 2: 12.3, ...}
```

### Request Format

Requests sent to the scheduler include:

```json
{
  "session_id": 12345,
  "prompt": "User's input text",
  "response": "Expected response from dataset",
  "is_last": false
}
```

- `session_id`: Unique identifier for the session
- `prompt`: The user's prompt/query
- `response`: The expected response (from dataset, for validation purposes)
- `is_last`: Boolean indicating if this is the last request in the session

### Stopping the Generator

To gracefully stop the workload generator:

1. Press `Ctrl+C` in the terminal
2. The server will stop and clean up connections

