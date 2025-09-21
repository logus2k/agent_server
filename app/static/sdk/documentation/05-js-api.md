# JS SDK API

```ts
class AgentClient {
	constructor(opts?: { url?: string; path?: string }); // default path: /socket.io
	connect(opts?: { onReconnect?(attempt:number):void }): Promise<void>;
	disconnect(): void;

	onStream(cbs?: {
		onStarted?(runId:string|null):void;
		onChunk?(piece:string):void;
		onText?(full:string):void;
		onDone?():void;
		onError?(e:{code:string; message:string; runId?:string|null}):void;
	}): void;

	runText(text: string, opts: { agent: string; threadId?: string }, cbs?: SameAsOnStream): Promise<{ runId:string|null; text:string }>;

	cancel(): void;

	sttSubscribe(args: { sttUrl: string; clientId: string; agent: string; threadId?: string }): Promise<void>;
	sttUnsubscribe(args: { sttUrl: string; clientId: string }): Promise<void>;

	get activeRunId(): string | null;
}
```

**Notes**

- `runText` sends `{ text, agent, thread_id }` to match server expectations.
- `sttSubscribe` sends `{ sttUrl, clientId, agent, threadId }` to match server's JoinSTT handler.
- Changing agents for STT requires re-subscribe.
