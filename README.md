# Agent Server: Local AI Orchestration Backend

The **Agent Server** is a critical backend component designed to manage and coordinate a full, local, and privacy-focused AI stack. It serves as the central control plane, receiving user input (often voice via STT), processing it through a Large Language Model (LLM), executing the resulting actions, and generating a verbal or textual response (via TTS).

It is built for high responsiveness and privacy, ensuring all natural language processing and command execution are performed entirely on the local system.

***

## Role and Core Functionality

The Agent Server acts as the **AI Orchestrator**, translating complex, natural language requests into structured, executable commands for any connected client application.

Its core functions include:

1.  **Command Interpretation (NLP):** Receives raw text input (e.g., from an STT service) and forwards it to the **Local LLM Backend** for deep semantic understanding.
2.  **Action Planning:** Determines the logical steps necessary to fulfill the user's request. This involves breaking down an instruction into a sequence of discrete, structured actions (e.g., a multi-step query, a data retrieval operation, or a client command).
3.  **Command Execution:** Executes internal logic, manages data flows, and prepares the output required by the client application.
4.  **Response Synthesis:** Formulates a coherent, natural language text response based on the action taken and the results obtained.
5.  **Output Delivery:** Sends structured commands (e.g., JSON payloads) back to the client application and sends the synthesized text response to a **TTS Server** for audible feedback.

***

## Local AI Pipeline

The Agent Server coordinates a multi-server, privacy-first AI pipeline that operates entirely on the local machine. This architecture allows for low-latency, secure interaction without relying on external cloud services. 

The standard flow is as follows:

1.  **Input:** Voice input is captured by the client and sent to the **STT Server**.
2.  **Transcription:** The STT Server converts the audio into a text command.
3.  **Agent Processing:** The Agent Server receives the text, engages the **Local LLM Backend** for interpretation, and plans the necessary action.
4.  **Client Command:** The Agent Server sends the structured command to the client (e.g., a web application) for execution.
5.  **Verbal Feedback:** The Agent Server generates a text response and sends it to the **TTS Server**.
6.  **Output:** The TTS Server converts the text into audio output for the user.

***

## Technical Dependencies

The Agent Server is a middleware component designed to connect and manage dedicated local microservices.

| Dependency | Purpose | Repository Link |
| :--- | :--- | :--- |
| **STT Server** | Provides the local **Speech-to-Text** service for transcribing voice input. | [github.com/logus2k/stt_server](https://github.com/logus2k/stt_server) |
| **TTS Server** | Provides the local **Text-to-Speech** service for generating audible responses. | [github.com/logus2k/tts_server](https://github.com/logus2k/tts_server) |
| **Local LLM Backend** | Runs a **Large Language Model** locally for natural language processing and complex command interpretation. | *(Utilizes a common local framework, e.g., Llama.cpp)* |
| **Communication Layer**| Uses local HTTP/WebSockets to ensure low-latency communication with the other services and the client application. | **Fast API** |

***

### License

This project's license is Apache 2.0.

---