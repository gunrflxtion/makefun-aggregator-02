// The client wires, the one upstream wire, and the model field.
//
// The env calls this server as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with
// server-sent events. The server dials the grader's gateway, an
// OpenAI-compatible endpoint, so every prompt goes upstream as a Chat
// Completions request and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body to the gateway and relays the
//     gateway's JSON or SSE answer, with every `model` field set to the id
//     this server dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system
//     prompt, image blocks, tool definitions, tool results and tool choice
//     included) and rewrites the gateway's answer into a Messages object, or
//     into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the
//     stop reason and the usage translated. Tool calls become tool_use blocks
//     and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really
//   answered, so a response's `model` must be the model this server dialled.
//   OpenRouter's own answer names a free model without the gateway's
//   `openrouter/` prefix; these functions put the dialled id back on every
//   answer and every streamed chunk. They never make a model up: the id came
//   from the brief and went to the gateway in the request's `model`.
//
// STREAMS ARE BUFFERED, THEN RELAYED
//   Unlike a pass-through relay, this server reads the gateway's whole SSE
//   stream before answering the client. That is what lets it check the
//   prompt's token in the provider's answer and fail over to another model
//   when the token is missing, instead of serving an answer the scorer
//   cannot credit. The client waits up to 30 seconds per prompt; a buffered
//   stream costs a few seconds. The event order the client sees is the
//   provider's, replayed in full.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same gateway answer give the same
// bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { estimateTokens, isRecord, type Json } from './brief.ts';

/** One model the server lists, as GET /v1/models names it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this server serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or inline image. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The searchable text of a Chat Completions answer: content plus tool args. */
export function textOfOpenAIAnswer(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const parts: string[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') parts.push(text);
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    if (typeof call.function.name === 'string') parts.push(call.function.name);
    if (typeof call.function.arguments === 'string') parts.push(call.function.arguments);
  }
  return parts.join('\n');
}

/**
 * Usage for an answer, the provider's own when it carries one, else an
 * estimate from the request and answer text: the wire checks want a whole
 * input count of at least 0 and an output count of at least 1, and a stream
 * or answer that omits usage must not fail on that alone.
 */
export function usageOf(
  body: Json,
  estimatedInput: number,
  estimatedOutput: number,
): { prompt_tokens: number; completion_tokens: number } {
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.prompt_tokens);
  const output = numberOrZero(usage.completion_tokens);
  return {
    prompt_tokens: input > 0 ? input : Math.max(0, estimatedInput),
    completion_tokens: output > 0 ? output : Math.max(1, estimatedOutput),
  };
}

/** A Chat Completions answer with `model` set to the id the server dialled. */
export function openAIJsonWithModel(body: Json, model: string, estimatedInput: number): Json {
  const usage = usageOf(body, estimatedInput, Math.max(1, estimateTokens(textOfOpenAIAnswer(body))));
  return { ...body, model, usage };
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string, estimatedInput: number): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = usageOf(body, estimatedInput, Math.max(1, estimateTokens(textOfOpenAIAnswer(body))));
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** One server-sent event: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a whole SSE body into frames, on the blank line that ends each. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** Read a whole upstream SSE body into frames. */
export async function readSseFrames(upstream: ReadableStream<Uint8Array>): Promise<SseFrame[]> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    }
    buffer += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return parseSseFrames(buffer);
}

/**
 * One finish chunk, nothing after it: keep the frames up to and including
 * the first chunk that carries a finish_reason, and drop the rest. Some
 * upstreams repeat the finish chunk (empty deltas, finish_reason "stop",
 * several times over) and may send content after it; the wire allows
 * exactly one finish chunk, and a content delta after it voids the stream.
 * Usage seen anywhere in the stream is carried onto the finish chunk when
 * it lacks one, so the real counts survive the cut.
 */
export function normalizeOpenAIStreamFrames(frames: readonly SseFrame[]): SseFrame[] {
  let finishAt = -1;
  let usage: Json | null = null;
  for (let i = 0; i < frames.length; i++) {
    const payload = frames[i].data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) usage = chunk.usage;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null && typeof choice.finish_reason === 'string' && finishAt === -1) {
      finishAt = i;
    }
  }
  if (finishAt === -1) return [...frames];
  const kept = frames.slice(0, finishAt + 1);
  if (usage !== null) {
    const last = kept[kept.length - 1];
    try {
      const chunk: unknown = JSON.parse(last.data.trim());
      if (isRecord(chunk) && !isRecord(chunk.usage)) {
        kept[kept.length - 1] = { ...last, data: JSON.stringify({ ...chunk, usage }) };
      }
    } catch {
      // The finish chunk is not JSON; leave it as it arrived.
    }
  }
  return kept;
}

/** The text a Chat Completions stream carried, and its usage and stop reason. */
export interface StreamDigest {
  text: string;
  toolNames: string[];
  stopReason: string | null;
  inputTokens: number;
  outputTokens: number;
  sawFinish: boolean;
  sawDone: boolean;
}

/** Digest buffered OpenAI SSE frames: text, tool names, usage, stop reason. */
export function digestOpenAIStream(frames: readonly SseFrame[]): StreamDigest {
  let text = '';
  const toolNames: string[] = [];
  let stopReason: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let sawFinish = false;
  let sawDone = false;
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      sawDone = true;
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      if (typeof call.function.name === 'string' && call.function.name !== '') {
        toolNames.push(call.function.name);
      }
      if (typeof call.function.arguments === 'string') text += call.function.arguments;
    }
    if (typeof choice.finish_reason === 'string') {
      sawFinish = true;
      stopReason = choice.finish_reason;
    }
  }
  return { text, toolNames, stopReason, inputTokens, outputTokens, sawFinish, sawDone };
}

/**
 * Buffered OpenAI SSE frames relayed to the client, every data payload's
 * `model` the id the server dialled. A stream that carried no usage gets it
 * on its finish chunk (the wire wants a whole input and an output of at
 * least 1); one with no finish chunk gets a synthetic one, and one with no
 * [DONE] gets it appended, so the answer is well-formed either way.
 */
export function relayOpenAIStream(
  frames: readonly SseFrame[],
  model: string,
  estimatedInput: number,
  estimatedOutput: number,
): string {
  const digest = digestOpenAIStream(frames);
  const out: string[] = [];
  let sawFinish = false;
  let sawDone = false;
  let usageWritten = digest.inputTokens > 0 || digest.outputTokens > 0;
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      sawDone = true;
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      out.push(sseEvent(frame.event, payload));
      continue;
    }
    if (!isRecord(chunk)) {
      out.push(sseEvent(frame.event, payload));
      continue;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    const isFinish = choice !== null && typeof choice.finish_reason === 'string';
    if (isFinish) sawFinish = true;
    const rewritten: Json = { ...chunk, model };
    if (isFinish && !usageWritten) {
      rewritten.usage = {
        prompt_tokens: Math.max(0, estimatedInput),
        completion_tokens: Math.max(1, estimatedOutput),
      };
      usageWritten = true;
    }
    out.push(sseEvent(frame.event, rewritten));
  }
  if (!sawFinish) {
    out.push(
      sseEvent(null, {
        id: `chatcmpl-finish`,
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: digest.stopReason ?? 'stop' }],
        usage: {
          prompt_tokens: Math.max(0, estimatedInput),
          completion_tokens: Math.max(1, estimatedOutput),
        },
      }),
    );
  }
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/**
 * Buffered OpenAI SSE frames as the Anthropic event sequence: message_start
 * on the first chunk, content_block_start/delta/stop per text block or tool
 * call, then message_delta with the stop reason and usage, then
 * message_stop. `model` is the id the server dialled, named back to the
 * client. Usage falls back to estimates when the stream carried none.
 */
export function anthropicEventsFromFrames(
  frames: readonly SseFrame[],
  model: string,
  estimatedInput: number,
): string {
  const digest = digestOpenAIStream(frames);
  const out: string[] = [];
  const outputTokens =
    digest.outputTokens > 0 ? digest.outputTokens : Math.max(1, estimateTokens(digest.text));
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: digest.inputTokens > 0 ? digest.inputTokens : Math.max(0, estimatedInput),
          output_tokens: 0,
        },
      },
    }),
  );
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  const closeText = (): void => {
    if (textIndex === -1) return;
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
    textIndex = -1;
  };
  const closeTools = (): void => {
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
  };
  let emittedAny = false;
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index: textIndex,
            content_block: { type: 'text', text: '' },
          }),
        );
      }
      out.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index: textIndex,
          delta: { type: 'text_delta', text: delta.content },
        }),
      );
      emittedAny = true;
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        closeText();
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(
          sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index,
            delta: { type: 'input_json_delta', partial_json: fn.arguments },
          }),
        );
        emittedAny = true;
      }
    }
  }
  if (!emittedAny) {
    textIndex = nextIndex;
    nextIndex += 1;
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: textIndex,
        content_block: { type: 'text', text: '' },
      }),
    );
  }
  closeText();
  closeTools();
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(digest.stopReason), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}
