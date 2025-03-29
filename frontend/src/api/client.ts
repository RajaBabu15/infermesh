export interface WorkerLoad {
  id: string;
  model: string;
  type: "openai" | "vllm";
  role: "prefill" | "decode" | "mixed";
  tokens_in_flight: number;
  cache_utilization: number;
  circuit_state: "CLOSED" | "OPEN" | "HALF_OPEN";
}

export interface SessionCounters {
  prefill_complete_received_total: number;
  sticky_hits_total: number;
  sticky_misses_total: number;
  session_eviction_total: number;
}

export interface RedisEventCounters {
  received_total: number;
  dropped_total: number;
  by_type: Record<string, number>;
  drops_by_reason: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  workers: number;
  worker_loads: WorkerLoad[];
  trie_entries: number;
  cache_hits_total: number;
  redis_enabled: boolean;
  redis_transport: "streams" | "pubsub" | null;
  disaggregation_enabled: boolean;
  prometheus_enabled: boolean;
  otel_enabled: boolean;
  active_sessions?: number;
  session_counters?: SessionCounters;
  redis_events?: RedisEventCounters;
}

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface ModelsResponse {
  object: string;
  data: ModelInfo[];
}

export interface ChatMessage {
  role: "user" | "assistant" | "system" | "tool" | "function";
  content: string | null;
  tool_calls?: any[];
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  model: string;
  choices: {
    index: number;
    message: ChatMessage;
    finish_reason: string | null;
  }[];
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

export interface ChatParams {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetch("/v1/models");
  if (!res.ok) throw new Error(`Models fetch failed: ${res.status}`);
  return res.json();
}

export async function chatCompletion(
  model: string,
  messages: ChatMessage[],
  stream: boolean,
  sessionId?: string,
  params?: ChatParams,
): Promise<Response> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (sessionId) headers["X-InferMesh-Session-ID"] = sessionId;
  return fetch("/v1/chat/completions", {
    method: "POST",
    headers,
    body: JSON.stringify({ model, messages, stream, ...params }),
  });
}
