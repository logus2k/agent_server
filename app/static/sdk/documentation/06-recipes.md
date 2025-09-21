# Recipes

## Send on Enter (Shift+Enter = newline)
```js
textarea.addEventListener("keydown", (e) => {
	if (!e.isComposing && e.key === "Enter" && !e.shiftKey) {
		e.preventDefault();
		send();
	}
});
```

## Switch agent while using STT
```js
await client.sttUnsubscribe({ sttUrl, clientId });
await client.sttSubscribe({ sttUrl, clientId, agent: newAgent, threadId });
```

## Fallback to text if STT is down
```js
try {
	await client.sttSubscribe({ sttUrl, clientId, agent, threadId });
} catch {
	console.warn("STT unavailable; using text only");
}
```

## Persist thread across reloads
```js
const key = "threadId";
let threadId = localStorage.getItem(key) ?? crypto.randomUUID();
localStorage.setItem(key, threadId);
```

## Cancel on route change
```js
window.addEventListener("beforeunload", () => client.cancel());
```
