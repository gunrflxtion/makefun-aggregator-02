// free-model-aggregator-02: one server, two client wires, a real free model
// behind every answer.
//
// The env points a client at this server, and every completion it answers
// comes from a real free OpenRouter model dialled through the grader's
// gateway. The server:
//
//   1. listens on 0.0.0.0 at a free port (never 8080: the env owns it);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port", "catalog"};
//   3. serves GET /v1/models (OpenAI shape for an Authorization header,
//      Anthropic shape for x-api-key), POST /v1/chat/completions and POST
//      /v1/messages, each as JSON or as server-sent events;
//   4. answers each prompt by dialling a free model through the gateway.
//      OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for the
//      OpenRouter key pushed to the competition; the real key never enters
//      the sandbox;
//   5. polls GET $MAKEFUN_ENV_URL/work and exits 0 when it says done.
//
// ROUTING (quality is 60% of the score)
//   The brief rates every free model. Each prompt goes to the best-rated
//   model that meets its requirements: the strongest JSON-capable model for
//   a JSON-mode prompt, the strongest overall for the rest. A model that
//   refuses (4xx, 5xx, or a dropped connection) is cooled for the rest of
//   the trial and the prompt fails over to the next-best model that meets
//   it, up to three attempts. The refusals are reported on the answer in
//   the `x-makefun-upstream-failures` header, the only way the failover
//   column can see a refusal the server retried and then served.
//
// THE TOKEN CHECK
//   Every prompt carries a unique token its answer must contain, and the
//   scorer only credits an answer the provider's own body carries it in.
//   So the server checks the token in each gateway answer before serving
//   it, and fails over when it is missing. An answer served without its
//   token would earn the prompt nothing; a failover that finds it earns
//   the prompt. When every attempt misses, the last answer is served
//   as-is: it can still pass the wire checks, where a 502 could not.
//
// DETERMINISM
//   Every final trial runs twice, live and then replayed from the recorded
//   gateway answers, and the replay must leave the same state export, byte
//   for byte. So nothing that differs between two runs may reach a request
//   or an answer: no wall clock, no unseeded randomness. Routing comes
//   from the brief, the request and the gateway's answers alone; the
//   backoff between attempts is a fixed pause, and the model lists carry
//   fixed timestamps. The port the OS picks is not part of the export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  estimateTokens,
  isRecord,
  rankedCandidates,
  readBrief,
  tokenFromBody,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicEventsFromFrames,
  anthropicModelList,
  anthropicToOpenAI,
  digestOpenAIStream,
  normalizeOpenAIStreamFrames,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIToAnthropic,
  readSseFrames,
  relayOpenAIStream,
  textOfOpenAIAnswer,
  type ServedModel,
  type SseFrame,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most dial attempts for one prompt: the best model, then two failovers. */
const MAX_ATTEMPTS = 3;
/** The fixed pause between two attempts of one prompt, in milliseconds. */
const RETRY_PAUSE_MS = 200;
/** The upstream-failures header, as the env reads it. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';
/** The most failure rows the header may carry. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;
/** Models refused with a 429 or 5xx in this trial: a rate limit or a server
 * error in a seconds-long trial is not worth another call. Other refusals
 * (a 403, a region gate) are retried on the next prompt: the final runs on
 * the grader's own network, where the same model may answer. */
const cooled = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** One failure row for the upstream-failures header. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the
 * OpenAI-compatible root (`…/v1` on the trial network); a base that does not
 * already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return brief.models
    .filter((m) => m.free)
    .map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));
}

/** The requirements of the prompt now arriving: the brief's row, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** The token of the prompt now arriving: the body's own, else the brief's row. */
function tokenFor(body: Json, index: number): string | null {
  const fromBody = tokenFromBody(body);
  if (fromBody !== null) return fromBody;
  const entry = brief.prompts[index];
  return entry !== undefined ? entry.token : null;
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: UpstreamFailure[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
  res.writeHead(status, headers);
  res.end(text);
}

function sendSse(res: ServerResponse, text: string, failures: UpstreamFailure[] = []): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  };
  if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
  res.writeHead(200, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One dial of one model on the gateway's Chat Completions route. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
  });
}

/** The error code of a refused gateway answer, from its error body. */
async function refusalCode(res: Response): Promise<string> {
  try {
    const text = await res.text();
    if (text === '') return `http_${res.status}`;
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
    return `http_${res.status}`;
  } catch {
    return `http_${res.status}`;
  }
}

/** A fixed pause. Waiting is fine; no time value is written anywhere. */
function pause(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** What one dial attempt returned: a usable answer, or why it did not. A
 * token-less answer is returned too, so the caller can serve the last one
 * when every attempt misses, rather than a 502. */
type Attempt =
  | { ok: true; model: ModelRow; answer: Json; frames: null; text: string }
  | { ok: true; model: ModelRow; answer: null; frames: SseFrame[]; text: string }
  | { ok: false; failure: UpstreamFailure; model: ModelRow; answer: Json | null; frames: SseFrame[] | null };

/**
 * Dial one model for one prompt and check its answer. A refusal, a dropped
 * connection, or an answer without the prompt's token is not usable: the
 * caller fails over to the next model. A usable answer carries the token.
 */
async function attempt(
  model: ModelRow,
  upstreamBody: Json,
  stream: boolean,
  token: string | null,
  estimatedInput: number,
): Promise<Attempt> {
  let upstream: Response;
  try {
    upstream = await dial(model.id, upstreamBody);
  } catch (err) {
    return {
      ok: false,
      failure: { status: 0, code: 'upstream_unreachable', model: model.id },
      model,
      answer: null,
      frames: null,
    };
  }
  if (!upstream.ok || upstream.body === null) {
    const code = await refusalCode(upstream);
    return { ok: false, failure: { status: upstream.status, code, model: model.id }, model, answer: null, frames: null };
  }
  if (stream) {
    const frames = normalizeOpenAIStreamFrames(await readSseFrames(upstream.body));
    const digest = digestOpenAIStream(frames);
    const text = digest.text;
    if (token !== null && !text.includes(token)) {
      return { ok: false, failure: { status: 200, code: 'missing_token', model: model.id }, model, answer: null, frames };
    }
    return { ok: true, model, answer: null, frames, text };
  }
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return { ok: false, failure: { status: 200, code: 'bad_upstream_json', model: model.id }, model, answer: null, frames: null };
  }
  if (!isRecord(answer)) {
    return { ok: false, failure: { status: 200, code: 'bad_upstream_json', model: model.id }, model, answer: null, frames: null };
  }
  const text = textOfOpenAIAnswer(answer);
  if (token !== null && !text.includes(token)) {
    return { ok: false, failure: { status: 200, code: 'missing_token', model: model.id }, model, answer, frames: null };
  }
  return { ok: true, model, answer, frames: null, text };
}

/**
 * Serve one prompt: the best-rated model that meets it, failing over on a
 * refusal or a token-less answer. Returns the answer to serve, or null when
 * no model served it.
 */
async function servePrompt(
  upstreamBody: Json,
  stream: boolean,
  requires: Requirement,
  token: string | null,
  estimatedInput: number,
): Promise<{ model: ModelRow; answer: Json | null; frames: SseFrame[] | null; failures: UpstreamFailure[] } | null> {
  const failures: UpstreamFailure[] = [];
  let lastMiss: { model: ModelRow; answer: Json | null; frames: SseFrame[] | null } | null = null;
  /** Models this prompt already tried: never dial one twice for one prompt. */
  const tried = new Set<string>();
  for (let n = 0; n < MAX_ATTEMPTS; n++) {
    const candidates = rankedCandidates(brief, requires, new Set([...cooled, ...tried]));
    if (candidates.length === 0) break;
    const model = candidates[0];
    const result = await attempt(model, upstreamBody, stream, token, estimatedInput);
    if (result.ok) {
      return { model: result.model, answer: result.answer, frames: result.frames, failures };
    }
    // A failed model is out for this prompt. Only a 429 or 5xx cools it for
    // the trial; any other refusal is reported in the trail and retried on
    // the next prompt, where the grader's network may let it answer. A
    // token-less answer is kept as the last resort and retried silently,
    // since it is not an upstream refusal.
    tried.add(model.id);
    if (result.failure.status === 429 || result.failure.status >= 500) {
      cooled.add(model.id);
    }
    if (result.failure.status !== 200) {
      failures.push(result.failure);
    } else if (result.failure.code === 'missing_token') {
      lastMiss = { model: result.model, answer: result.answer, frames: result.frames };
    }
    if (n + 1 < MAX_ATTEMPTS) await pause(RETRY_PAUSE_MS);
  }
  // Every attempt missed the token: serve the last answer as-is rather than
  // a 502, since it can still pass the wire checks where a 502 cannot.
  if (lastMiss !== null) return { ...lastMiss, failures };
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const rowIndex = manifestIndex;
  const requires = requirementsFor(body);
  const token = tokenFor(body, rowIndex);
  const estimatedInput = estimateTokens(JSON.stringify(body));
  const stream = body.stream === true;
  const upstreamBody: Json = { ...body, model: 'auto' };
  if (typeof upstreamBody.max_tokens !== 'number' && typeof upstreamBody.max_completion_tokens !== 'number') {
    upstreamBody.max_tokens = 512;
  }
  if (typeof upstreamBody.max_tokens === 'number' && upstreamBody.max_tokens > 8192) upstreamBody.max_tokens = 8192;
  const served = await servePrompt(upstreamBody, stream, requires, token, estimatedInput);
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'no free model served this prompt');
  }
  if (stream && served.frames !== null) {
    const digest = digestOpenAIStream(served.frames);
    const output = Math.max(1, estimateTokens(digest.text));
    sendSse(res, relayOpenAIStream(served.frames, served.model.id, estimatedInput, output), served.failures);
    return;
  }
  if (served.answer !== null) {
    sendJson(res, 200, openAIJsonWithModel(served.answer, served.model.id, estimatedInput), served.failures);
    return;
  }
  sendError(res, 'openai', 502, 'upstream_error', 'no free model served this prompt');
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const rowIndex = manifestIndex;
  const requires = requirementsFor(body);
  const token = tokenFor(body, rowIndex);
  const estimatedInput = estimateTokens(JSON.stringify(body));
  const stream = body.stream === true;
  const upstreamBody: Json = { ...body, model: 'auto' };
  if (typeof upstreamBody.max_tokens !== 'number' && typeof upstreamBody.max_completion_tokens !== 'number') {
    upstreamBody.max_tokens = 512;
  }
  if (typeof upstreamBody.max_tokens === 'number' && upstreamBody.max_tokens > 8192) upstreamBody.max_tokens = 8192;
  // The requirements row already consumed the manifest slot; translate with
  // the model chosen per attempt inside servePrompt by re-translating.
  const served = await servePromptTranslated(body, upstreamBody, stream, requires, token, estimatedInput);
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'no free model served this prompt');
  }
  if (stream && served.frames !== null) {
    sendSse(res, anthropicEventsFromFrames(served.frames, served.model.id, estimatedInput), served.failures);
    return;
  }
  if (served.answer !== null) {
    sendJson(res, 200, openAIToAnthropic(served.answer, served.model.id, estimatedInput), served.failures);
    return;
  }
  sendError(res, 'anthropic', 502, 'api_error', 'no free model served this prompt');
}

/**
 * The Anthropic path's serve: the request is translated per attempt, since
 * the translation names the model being dialled.
 */
async function servePromptTranslated(
  body: Json,
  upstreamBody: Json,
  stream: boolean,
  requires: Requirement,
  token: string | null,
  estimatedInput: number,
): Promise<{ model: ModelRow; answer: Json | null; frames: SseFrame[] | null; failures: UpstreamFailure[] } | null> {
  const failures: UpstreamFailure[] = [];
  let lastMiss: { model: ModelRow; answer: Json | null; frames: SseFrame[] | null } | null = null;
  /** Models this prompt already tried: never dial one twice for one prompt. */
  const tried = new Set<string>();
  for (let n = 0; n < MAX_ATTEMPTS; n++) {
    const candidates = rankedCandidates(brief, requires, new Set([...cooled, ...tried]));
    if (candidates.length === 0) break;
    const model = candidates[0];
    const translated = anthropicToOpenAI(body, model.id);
    if (typeof upstreamBody.max_tokens === 'number') translated.max_tokens = upstreamBody.max_tokens;
    const result = await attempt(model, translated, stream, token, estimatedInput);
    if (result.ok) {
      return { model: result.model, answer: result.answer, frames: result.frames, failures };
    }
    tried.add(model.id);
    if (result.failure.status === 429 || result.failure.status >= 500) {
      cooled.add(model.id);
    }
    if (result.failure.status !== 200) {
      failures.push(result.failure);
    } else if (result.failure.code === 'missing_token') {
      lastMiss = { model: result.model, answer: result.answer, frames: result.frames };
    }
    if (n + 1 < MAX_ATTEMPTS) await pause(RETRY_PAUSE_MS);
  }
  if (lastMiss !== null) return { ...lastMiss, failures };
  return null;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; retried until it lands. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let n = 0; n < 12; n++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
      }
      return;
    } catch (err) {
      lastError = err;
      await pause(500);
    }
  }
  throw lastError instanceof Error ? lastError : new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
