/**
 * Zod schema for the openrouter_chat tool.
 * Defines input validation for chat completion requests.
 */

import { z } from 'zod';

/**
 * Schema for chat message roles
 */
export const MessageRoleEnum = z.enum(['system', 'user', 'assistant', 'tool']);

/**
 * Schema for a content part (text or image_url) for multimodal messages
 */
export const ContentPartSchema = z.union([
  z.object({ type: z.literal('text'), text: z.string() }),
  z.object({
    type: z.literal('image_url'),
    image_url: z.object({
      url: z.string(),
      detail: z.string().optional(),
    }),
  }),
]);

export type ContentPart = z.infer<typeof ContentPartSchema>;

/**
 * Schema for message content — string or array of content parts (for multimodal/vision)
 */
export const MessageContentSchema = z.union([z.string(), z.array(ContentPartSchema)]);

/**
 * Schema for a single chat message
 */
export const ChatMessageSchema = z.object({
  role: MessageRoleEnum,
  content: MessageContentSchema.describe('Message content. String for text, or array of content parts for multimodal (text + images).'),
  name: z.string().optional(),
  tool_call_id: z.string().optional(),
});

export type ChatMessage = z.infer<typeof ChatMessageSchema>;

/**
 * Schema for function definition in tool calls
 */
export const FunctionDefinitionSchema = z.object({
  name: z.string().min(1, 'Function name is required'),
  description: z.string().optional(),
  parameters: z.record(z.unknown()).optional(),
});

/**
 * Schema for a tool definition (OpenAI-compatible function tool)
 */
export const FunctionToolSchema = z.object({
  type: z.literal('function'),
  function: FunctionDefinitionSchema,
});

/**
 * Schema for OpenRouter server tools (e.g., openrouter:web_search).
 * These are OpenRouter-native tools that are passed through to the API
 * without requiring function definitions.
 * @see https://openrouter.ai/docs/guides/features/server-tools/web-search
 */
export const ServerToolSchema = z.object({
  type: z.custom<`openrouter:${string}`>(
    (val) => typeof val === 'string' && /^openrouter:.+$/.test(val),
    'Server tool type must start with "openrouter:" and include a tool name',
  ),
  parameters: z.object({
    engine: z.enum(['auto', 'native', 'exa', 'firecrawl', 'parallel']).optional(),
    max_results: z.number().int().min(1).max(25).optional(),
    max_total_results: z.number().int().positive().optional(),
    search_context_size: z.enum(['low', 'medium', 'high']).optional(),
    user_location: z.object({
      type: z.literal('approximate').optional(),
      city: z.string().optional(),
      region: z.string().optional(),
      country: z.string().optional(),
      timezone: z.string().optional(),
    }).optional(),
    allowed_domains: z.array(z.string()).optional(),
    excluded_domains: z.array(z.string()).optional(),
  }).optional(),
});

/**
 * Schema for a tool definition — either an OpenAI-compatible function tool
 * or an OpenRouter server tool (e.g., openrouter:web_search).
 */
export const ToolDefinitionSchema = z.union([FunctionToolSchema, ServerToolSchema]);

export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;

/**
 * Schema for tool_choice parameter
 */
export const ToolChoiceSchema = z.union([
  z.literal('auto'),
  z.literal('none'),
  z.literal('required'),
  z.object({
    type: z.literal('function'),
    function: z.object({
      name: z.string().min(1),
    }),
  }),
]);

export type ToolChoice = z.infer<typeof ToolChoiceSchema>;

/**
 * Schema for response_format parameter (text, json_object, or json_schema)
 */
export const ResponseFormatSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('text') }),
  z.object({ type: z.literal('json_object') }),
  z.object({
    type: z.literal('json_schema'),
    json_schema: z.object({
      name: z.string(),
      strict: z.boolean().optional(),
      schema: z.record(z.unknown()),
    }),
  }),
]);

export type ResponseFormat = z.infer<typeof ResponseFormatSchema>;

/**
 * Base input schema for the chat tool (before refinement)
 */
const ChatInputBaseSchema = z.object({
  /** Model ID to use for the chat completion (required) */
  model: z
    .string()
    .min(1, 'Model ID is required')
    .describe('The model ID (format: "provider/model-name"). MUST be discovered via openrouter_search_models or openrouter_list_models first - never guess from memory.'),

  /** System prompt / role description (convenience param) */
  role: z
    .string()
    .optional()
    .describe('System prompt / role description for the model (e.g. "You are a helpful coding assistant"). Creates a system message automatically.'),

  /** User message content (convenience param) */
  message: z
    .union([z.string(), z.array(ContentPartSchema)])
    .optional()
    .describe('The user message content. Can be a string or an array of content parts (text + images). Use this instead of the messages array for simple single-turn requests.'),

  /** Array of messages for the conversation */
  messages: z
    .array(ChatMessageSchema)
    .optional()
    .describe('Array of messages in the conversation. Not required if using the message param.'),

  /** Optional session ID to continue an existing session */
  session_id: z
    .string()
    .optional()
    .describe('Session ID to continue an existing conversation'),

  /** Whether to stream the response (default: true) */
  stream: z
    .boolean()
    .optional()
    .default(true)
    .describe('Whether to stream the response (default: true)'),

  /** Temperature for response randomness (0-2) */
  temperature: z
    .number()
    .min(0)
    .max(2)
    .optional()
    .describe('Temperature for response randomness (0-2)'),

  /** Maximum tokens to generate */
  max_tokens: z
    .number()
    .int()
    .positive()
    .optional()
    .default(16000)
    .describe('Maximum number of tokens to generate (default: 16000). Rarely needs to be changed — only lower this if you specifically need shorter responses.'),

  /** Array of tool definitions for function calling */
  tools: z
    .array(ToolDefinitionSchema)
    .optional()
    .describe('Array of tool definitions. Supports OpenAI-compatible function tools and OpenRouter server tools (e.g., {"type": "openrouter:web_search"}).'),

  /** Tool choice parameter */
  tool_choice: ToolChoiceSchema
    .optional()
    .describe('How to select which tool to call (auto, none, required, or specific function)'),

  /** Response format specification */
  response_format: ResponseFormatSchema
    .optional()
    .describe('Structured output format specification (text, json_object, or json_schema)'),

  /** Nucleus sampling threshold */
  top_p: z.number().min(0).max(1).optional()
    .describe('Nucleus sampling: top_p (0-1]'),

  /** Top-k sampling */
  top_k: z.number().int().min(0).optional()
    .describe('Top-k sampling: limits token selection at each step'),

  /** Minimum probability threshold */
  min_p: z.number().min(0).max(1).optional()
    .describe('Minimum probability relative to most likely token (0-1)'),

  /** Top-a dynamic filtering */
  top_a: z.number().min(0).max(1).optional()
    .describe('Dynamic filtering for sufficiently high probability tokens (0-1)'),

  /** Frequency penalty */
  frequency_penalty: z.number().min(-2).max(2).optional()
    .describe('Frequency penalty to reduce repetition (-2 to 2)'),

  /** Presence penalty */
  presence_penalty: z.number().min(-2).max(2).optional()
    .describe('Presence penalty to reduce repetition (-2 to 2)'),

  /** Repetition penalty */
  repetition_penalty: z.number().min(0).max(2).optional()
    .describe('Repetition penalty (0-2, 1.0 = no penalty)'),

  /** Seed for deterministic sampling */
  seed: z.number().int().optional()
    .describe('Seed for deterministic/reproducible sampling'),

  /** Stop sequences */
  stop: z.union([z.string(), z.array(z.string())]).optional()
    .describe('Stop sequence(s) to end generation'),

  /** Parallel tool calls */
  parallel_tool_calls: z.boolean().optional()
    .describe('Allow model to make multiple tool calls simultaneously'),

  /** Structured outputs */
  structured_outputs: z.boolean().optional()
    .describe('Enable structured JSON output mode'),

  /** Reasoning configuration */
  reasoning: z.object({
    effort: z.enum(['xhigh', 'high', 'medium', 'low', 'minimal', 'none']).optional(),
    max_tokens: z.number().int().positive().optional(),
    exclude: z.boolean().optional(),
    enabled: z.boolean().optional(),
  }).optional()
    .describe('Reasoning/thinking token configuration. Use effort OR max_tokens, not both.'),

  /** Plugins */
  plugins: z.array(z.object({
    id: z.string(),
    engine: z.enum(['native', 'exa']).optional(),
    max_results: z.number().int().optional(),
    search_prompt: z.string().optional(),
    enabled: z.boolean().optional(),
  })).optional()
    .describe('Plugins to extend model capabilities (web search, PDF parsing, response healing)'),

  /** Provider routing preferences */
  provider: z.object({
    order: z.array(z.string()).optional(),
    allow_fallbacks: z.boolean().optional(),
    require_parameters: z.boolean().optional(),
    data_collection: z.enum(['allow', 'deny']).optional(),
    ignore: z.array(z.string()).optional(),
    only: z.array(z.string()).optional(),
    quantizations: z.array(z.string()).optional(),
    sort: z.union([z.string(), z.object({ by: z.string(), partition: z.string().optional() })]).optional(),
  }).optional()
    .describe('Provider routing preferences (order, fallbacks, data collection, quantizations)'),

  /** Prompt transforms */
  transforms: z.array(z.string()).optional()
    .describe('Prompt transforms (e.g., ["middle-out"] for context compression)'),

  /** Fallback models */
  models: z.array(z.string()).optional()
    .describe('List of fallback model IDs to try if primary model fails'),

  /** Routing strategy */
  route: z.literal('fallback').optional()
    .describe('Routing strategy for fallback models'),

  /** Predicted output for latency reduction */
  prediction: z.object({
    type: z.literal('content'),
    content: z.string(),
  }).optional()
    .describe('Predicted output to reduce latency'),

  /** Verbosity constraint */
  verbosity: z.enum(['low', 'medium', 'high', 'max']).optional()
    .describe('Constrain response verbosity level'),

  /** Log probabilities */
  logprobs: z.boolean().optional()
    .describe('Return token log probabilities'),

  /** Top logprobs per token */
  top_logprobs: z.number().int().min(0).max(20).optional()
    .describe('Number of top logprobs per token (0-20)'),

  /** Logit bias */
  logit_bias: z.record(z.string(), z.number()).optional()
    .describe('Bias specific tokens by token ID'),

  /** Max completion tokens (newer alternative to max_tokens) */
  max_completion_tokens: z.number().int().positive().optional()
    .describe('Maximum completion tokens (newer alternative to max_tokens)'),

  /** End-user identifier */
  user: z.string().optional()
    .describe('Stable end-user identifier for abuse detection'),

  /** Debug options */
  debug: z.object({
    echo_upstream_body: z.boolean().optional(),
  }).optional()
    .describe('Debug options (e.g., echo the upstream request body)'),
});

/**
 * Input schema with validation: at least one of `message` or `messages` must be provided
 */
export const ChatInputSchema = ChatInputBaseSchema.refine(
  (data) => data.message !== undefined || (data.messages !== undefined && data.messages.length > 0),
  {
    message: 'At least one of "message" or "messages" must be provided',
    path: ['message'],
  }
);

export type ChatInput = z.infer<typeof ChatInputSchema>;

/**
 * Tool call structure in the response
 */
export interface ChatToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

/**
 * Usage statistics in the response
 */
export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/**
 * Rate limit status in the response
 */
export interface ChatRateLimitStatus {
  requestsRemaining?: number;
  requestsLimit?: number;
  tokensRemaining?: number;
  tokensLimit?: number;
  isApproachingLimit: boolean;
}

/**
 * Logprobs token entry
 */
export interface LogprobEntry {
  token: string;
  logprob: number;
  bytes?: number[];
  top_logprobs?: Array<{
    token: string;
    logprob: number;
    bytes?: number[];
  }>;
}

/**
 * Response structure for the chat tool
 */
export interface ChatResponse {
  /** The text content of the response */
  content: string | null;

  /** Tool calls requested by the model (if any) */
  tool_calls?: ChatToolCall[];

  /** Token usage statistics */
  usage?: ChatUsage;

  /** Session ID for continuing the conversation */
  session_id: string;

  /** Rate limit status */
  rate_limit_status?: ChatRateLimitStatus;

  /** Finish reason from the model */
  finish_reason?: string | null;

  /** Model ID used */
  model: string;

  /** Reasoning content from the model */
  reasoning?: string;

  /** Detailed reasoning steps */
  reasoning_details?: Array<{ type: string; summary?: string; text?: string; data?: string }>;

  /** Annotations (e.g., URL citations from web search plugins) */
  annotations?: Array<{ type: string; url_citation?: { url: string; title: string; content?: string } }>;

  /** Native finish reason from the upstream provider */
  native_finish_reason?: string;

  /** Token log probabilities */
  logprobs?: LogprobEntry[];
}

/**
 * Streaming chunk structure
 */
export interface StreamingChunk {
  /** Chunk index */
  index: number;

  /** Delta content for this chunk */
  delta: {
    content?: string;
    tool_calls?: Array<{
      index: number;
      id?: string;
      type?: 'function';
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
  };

  /** Finish reason (only in final chunk) */
  finish_reason?: string | null;
}

export default ChatInputSchema;
