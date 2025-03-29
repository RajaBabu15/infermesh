import { useState, useRef, useEffect } from "react";
import { useModels } from "../hooks/useModels";
import { chatCompletion } from "../api/client";
import type { ChatMessage, ChatParams } from "../api/client";

interface Turn {
  role: "user" | "assistant";
  content: string;
}

function renderContent(content: string | null, toolCalls?: any[]): React.ReactNode {
  if (content) return content;
  if (toolCalls?.length) return <span className="opacity-60 italic">tool call: {toolCalls[0]?.function?.name ?? "…"}</span>;
  return <span className="opacity-40 italic">thinking…</span>;
}

export default function Playground() {
  const { data: modelsData } = useModels();
  const models = modelsData?.data ?? [];

  const [model, setModel] = useState("");
  const [stream, setStream] = useState(false);
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);

  // Inference params
  const [temperature, setTemperature] = useState<string>("");
  const [maxTokens, setMaxTokens] = useState<string>("");
  const [topP, setTopP] = useState<string>("");
  const [showParams, setShowParams] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (models.length && !model) setModel(models[0].id);
  }, [models, model]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  function buildParams(): ChatParams {
    const p: ChatParams = {};
    const t = parseFloat(temperature);
    const m = parseInt(maxTokens, 10);
    const tp = parseFloat(topP);
    if (!isNaN(t)) p.temperature = t;
    if (!isNaN(m) && m > 0) p.max_tokens = m;
    if (!isNaN(tp)) p.top_p = tp;
    return p;
  }

  async function send() {
    if (!input.trim() || loading) return;
    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    const history: ChatMessage[] = turns.map((t) => ({ role: t.role, content: t.content }));

    setTurns((p) => [...p, { role: "user", content: userMsg.content! }]);
    setInput("");
    setLoading(true);
    setError("");

    try {
      const res = await chatCompletion(model, [...history, userMsg], stream, sessionId, buildParams());

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status} — ${body}`);
      }

      const newSid = res.headers.get("X-InferMesh-Session-ID");
      if (newSid) setSessionId(newSid);

      if (stream) {
        setTurns((p) => [...p, { role: "assistant", content: "" }]);
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6).trim();
            if (payload === "[DONE]") break;
            try {
              const chunk = JSON.parse(payload);
              const delta = chunk.choices?.[0]?.delta?.content ?? "";
              if (delta) {
                setTurns((p) => {
                  const updated = [...p];
                  updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    content: updated[updated.length - 1].content + delta,
                  };
                  return updated;
                });
              }
            } catch {
              // ignore malformed SSE chunks
            }
          }
        }
      } else {
        const data = await res.json();
        const msg = data.choices?.[0]?.message;
        const content = msg?.content ?? (msg?.tool_calls ? `[tool: ${msg.tool_calls[0]?.function?.name ?? "call"}]` : "(empty)");
        setTurns((p) => [...p, { role: "assistant", content }]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Toolbar */}
      <div className="flex items-center gap-4 mb-4 flex-wrap">
        <h1 className="text-xl font-semibold">Playground</h1>

        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="bg-gray-800 border border-gray-700 text-sm rounded-lg px-3 py-1.5 text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
        >
          {models.length === 0 && <option value="">Loading models…</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id}>{m.id}</option>
          ))}
        </select>

        <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={stream}
            onChange={(e) => setStream(e.target.checked)}
            className="rounded"
          />
          Stream
        </label>

        <button
          onClick={() => setShowParams((p) => !p)}
          className={`text-xs px-2.5 py-1 rounded-lg border transition-colors ${
            showParams
              ? "border-indigo-600 text-indigo-400 bg-indigo-900/20"
              : "border-gray-700 text-gray-500 hover:text-gray-300"
          }`}
        >
          Params
        </button>

        {turns.length > 0 && (
          <button
            onClick={() => { setTurns([]); setSessionId(undefined); }}
            className="ml-auto text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            Clear
          </button>
        )}
        {sessionId && (
          <span className="font-mono text-xs text-gray-600 truncate max-w-[160px]" title={sessionId}>
            sid: {sessionId.slice(-8)}
          </span>
        )}
      </div>

      {/* Parameter controls */}
      {showParams && (
        <div className="flex items-center gap-4 mb-3 flex-wrap bg-gray-900 border border-gray-800 rounded-xl px-4 py-3">
          {[
            { label: "Temperature", value: temperature, set: setTemperature, placeholder: "0.0 – 2.0" },
            { label: "Max tokens", value: maxTokens, set: setMaxTokens, placeholder: "e.g. 512" },
            { label: "Top-p", value: topP, set: setTopP, placeholder: "0.0 – 1.0" },
          ].map(({ label, value, set, placeholder }) => (
            <label key={label} className="flex items-center gap-2 text-xs text-gray-400">
              <span className="w-20 shrink-0">{label}</span>
              <input
                type="number"
                value={value}
                onChange={(e) => set(e.target.value)}
                placeholder={placeholder}
                className="w-28 bg-gray-800 border border-gray-700 rounded-lg px-2 py-1 text-white text-xs placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </label>
          ))}
          <button
            onClick={() => { setTemperature(""); setMaxTokens(""); setTopP(""); }}
            className="text-xs text-gray-600 hover:text-gray-400 transition-colors ml-auto"
          >
            Reset
          </button>
        </div>
      )}

      {/* Chat history */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {turns.length === 0 && (
          <p className="text-gray-600 text-sm text-center mt-16">
            Send a message to test the gateway.
          </p>
        )}
        {turns.map((turn, i) => (
          <div key={i} className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap leading-relaxed ${
                turn.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              {turn.content || <span className="opacity-40 italic">thinking…</span>}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {error && (
        <div className="mt-2 text-xs text-red-400 bg-red-900/20 border border-red-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {/* Input */}
      <div className="mt-4 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Message (Enter to send, Shift+Enter for newline)"
          rows={2}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="px-5 rounded-xl bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
