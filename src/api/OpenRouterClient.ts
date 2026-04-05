/**
 * OpenRouter API Client
 * Provides authenticated access to OpenRouter's unified API.
 */

import { Logger, logger as defaultLogger } from '../utils/logger.js';
import { CacheManager } from './CacheManager.js';
import { RateLimitManager, RateLimitInfo, ThrottleStatus } from './RateLimitManager.js';
import {
  ApiError,
  AuthError,
  ErrorCode,
  parseOpenRouterError,
  createNetworkError,
  OpenRouterErrorResponse,
} from './errors.js';

// ============================================================================
// Types
// ============================================================================

export interface OpenRouterClientConfig {
  apiKey: string;
  baseUrl?: string;
  httpReferer?: string;
  timeout?: number;
  requestsPerSecond?: number;
  logger?: Logger;
}

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body?: unknown;
  headers?: Record<string, string>;
  timeout?: number;
  skipCache?: boolean;
  skipThrottle?: boolean;
}

export interface ApiResponse<T> {
  data: T;
  rateLimits: RateLimitInfo | null;
  throttleStatus: ThrottleStatus;
  cached: boolean;
}

// Model types from OpenRouter API
export interface OpenRouterModel {
  id: string;
  name?: string;
  canonical_slug?: string;
  description?: string;
  created?: number;
  context_length?: number;
  hugging_face_id?: string;
  expiration_date?: string;
  pricing?: {
    prompt?: string;
    completion?: string;
    request?: string;
    image?: string;
    image_token?: string;
    image_output?: string;
    audio?: string;
    audio_output?: string;
    input_audio_cache?: string;
    web_search?: string;
    internal_reasoning?: string;
    input_cache_read?: string;
    input_cache_write?: string;
    discount?: number;
  };
  top_provider?: {
    context_length?: number;
    max_completion_tokens?: number;
    is_moderated?: boolean;
  };
  architecture?: {
    modality?: string;
    tokenizer?: string;
    instruct_type?: string;
    input_modalities?: string[];
    output_modalities?: string[];
  };
  supported_parameters?: string[];
  per_request_limits?: {
    prompt_tokens?: number;
    completion_tokens?: number;
  } | null;
  default_parameters?: {
    temperature?: number | null;
    top_p?: number | null;
    frequency_penalty?: number | null;
  };
}

export interface ModelsListResponse {
  data: OpenRouterModel[];
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | ContentPart[];
  name?: string;
  tool_call_id?: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  top_a?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  repetition_penalty?: number;
  seed?: number;
  stop?: string | string[];
  tools?: Array<
    | {
        type: 'function';
        function: {
          name: string;
          description?: string;
          parameters?: Record<string, unknown>;
        };
      }
    | {
        type: string; // OpenRouter server tools (e.g., "openrouter:web_search")
        parameters?: Record<string, unknown>;
      }
  >;
  tool_choice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };
  parallel_tool_calls?: boolean;
  response_format?: { type: 'text' | 'json_object' | 'json_schema'; json_schema?: { name: string; strict?: boolean; schema: Record<string, unknown> } };
  structured_outputs?: boolean;
  reasoning?: {
    effort?: 'xhigh' | 'high' | 'medium' | 'low' | 'minimal' | 'none';
    max_tokens?: number;
    exclude?: boolean;
    enabled?: boolean;
  };
  plugins?: Array<{
    id: string;
    engine?: 'native' | 'exa';
    max_results?: number;
    search_prompt?: string;
    enabled?: boolean;
  }>;
  provider?: {
    order?: string[];
    allow_fallbacks?: boolean;
    require_parameters?: boolean;
    data_collection?: 'allow' | 'deny';
    ignore?: string[];
    only?: string[];
    quantizations?: string[];
    sort?: string | { by: string; partition?: string };
  };
  transforms?: string[];
  models?: string[];
  route?: 'fallback';
  prediction?: { type: 'content'; content: string };
  usage?: { include: boolean };
  verbosity?: 'low' | 'medium' | 'high' | 'max';
  logprobs?: boolean;
  top_logprobs?: number;
  logit_bias?: Record<string, number>;
  max_completion_tokens?: number;
  user?: string;
  debug?: { echo_upstream_body?: boolean };
}

export interface ReasoningDetail {
  type: 'reasoning.summary' | 'reasoning.encrypted' | 'reasoning.text';
  id?: string | null;
  format?: string;
  index?: number;
  summary?: string;
  data?: string;
  text?: string;
  signature?: string | null;
}

export interface ChatCompletionResponse {
  id: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string | null;
      reasoning?: string;
      reasoning_details?: ReasoningDetail[];
      tool_calls?: Array<{
        id: string;
        type: string;
        function: {
          name: string;
          arguments: string;
        };
      }>;
      annotations?: Array<{
        type: string;
        url_citation?: {
          url: string;
          title: string;
          content?: string;
          start_index?: number;
          end_index?: number;
        };
      }>;
    };
    finish_reason: string | null;
    native_finish_reason?: string;
    logprobs?: {
      content?: Array<{
        token: string;
        logprob: number;
        bytes?: number[];
        top_logprobs?: Array<{
          token: string;
          logprob: number;
          bytes?: number[];
        }>;
      }>;
    } | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    cost?: number;
    prompt_tokens_details?: {
      cached_tokens?: number;
      audio_tokens?: number;
    };
    completion_tokens_details?: {
      reasoning_tokens?: number;
    };
  };
}

// Image generation types (via chat/completions with modalities)
// Content can be text or image_url type
export interface ContentPart {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: {
    url: string; // Base64 data URL (data:image/png;base64,...) or regular URL
  };
}

export interface ImageGenerationRequest {
  model: string;
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: ContentPart[]; // Must be array of content parts for image generation
  }>;
  modalities: ['text', 'image']; // Order matters: text first, then image
  stream?: boolean;
  image_config?: {
    aspect_ratio?: '1:1' | '2:3' | '3:2' | '3:4' | '4:3' | '4:5' | '5:4' | '9:16' | '16:9' | '21:9';
    image_size?: '1K' | '2K' | '4K';
  };
}

// Image object in the images array format
export interface ImageObject {
  type?: string;
  image_url?: {
    url: string;
  };
  imageUrl?: {
    url: string;
  };
}

export interface ImageGenerationResponse {
  id?: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      // Content is an array of content parts (text and image_url objects)
      content?: ContentPart[] | string | null;
      // Some models return images in a separate images array
      images?: ImageObject[];
    };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    cost?: number;
  };
}

// Credits types (from /api/v1/key endpoint)
export interface CreditsResponse {
  data: {
    limit: number | null;
    limit_remaining: number | null;
    usage: number;
    usage_daily: number;
    usage_weekly: number;
    usage_monthly: number;
    is_free_tier: boolean;
    rate_limit?: {
      requests: number;
      interval: string;
    };
  };
}

// Generation stats types
export interface GenerationStatsResponse {
  data: {
    id: string;
    model: string;
    provider_name?: string;
    upstream_id?: string;
    created_at: string;
    tokens_prompt: number;
    tokens_completion: number;
    native_tokens_prompt?: number;
    native_tokens_completion?: number;
    native_tokens_reasoning?: number;
    native_tokens_cached?: number;
    native_tokens_images?: number;
    num_media_prompt?: number;
    num_media_completion?: number;
    num_search_results?: number;
    total_cost: number;
    usage?: number;
    cache_discount?: number | null;
    upstream_inference_cost?: number;
    latency: number;
    generation_time: number;
    moderation_latency?: number;
    streamed: boolean;
    cancelled?: boolean;
    finish_reason: string;
    native_finish_reason?: string;
    is_byok?: boolean;
    origin?: string;
  };
}

// Model endpoints types
export interface ModelEndpoint {
  name: string;
  model_id: string;
  model_name?: string;
  provider_name: string;
  tag?: string;
  status?: string;
  context_length: number;
  max_completion_tokens?: number;
  max_prompt_tokens?: number;
  supported_parameters?: string[];
  quantization?: string;
  supports_implicit_caching?: boolean;
  pricing: {
    prompt?: string;
    completion?: string;
    request?: string;
    image?: string;
    web_search?: string;
    internal_reasoning?: string;
    input_cache_read?: string;
    input_cache_write?: string;
  };
  latency_last_30m?: {
    p50?: number;
    p75?: number;
    p90?: number;
    p99?: number;
  };
  throughput_last_30m?: {
    p50?: number;
    p75?: number;
    p90?: number;
    p99?: number;
  };
  uptime_last_30m?: number;
}

export interface ModelEndpointsResponse {
  data: {
    id: string;
    name: string;
    created?: number;
    description?: string;
    architecture?: {
      input_modalities?: string[];
      output_modalities?: string[];
      tokenizer?: string;
      instruct_type?: string;
    };
    endpoints: ModelEndpoint[];
  };
}

// ============================================================================
// OpenRouter Client Implementation
// ============================================================================

export class OpenRouterClient {
  private readonly config: Required<Omit<OpenRouterClientConfig, 'logger'>> & { logger: Logger };
  private readonly cacheManager: CacheManager;
  private readonly rateLimitManager: RateLimitManager;

  static readonly DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1';
  static readonly DEFAULT_TIMEOUT = 30000; // 30 seconds
  static readonly DEFAULT_RPS = 10;

  constructor(config: OpenRouterClientConfig) {
    if (!config.apiKey) {
      throw new AuthError('API key is required', ErrorCode.AUTH_MISSING_KEY);
    }

    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl ?? OpenRouterClient.DEFAULT_BASE_URL,
      httpReferer: config.httpReferer ?? 'https://github.com/openrouter-mcp-server',
      timeout: config.timeout ?? OpenRouterClient.DEFAULT_TIMEOUT,
      requestsPerSecond: config.requestsPerSecond ?? OpenRouterClient.DEFAULT_RPS,
      logger: config.logger ?? defaultLogger.child('api-client'),
    };

    this.cacheManager = new CacheManager({
      logger: this.config.logger.child('cache'),
    });

    this.rateLimitManager = new RateLimitManager({
      requestsPerSecond: this.config.requestsPerSecond,
      logger: this.config.logger.child('rate-limit'),
    });

    this.config.logger.debug('OpenRouterClient initialized', {
      baseUrl: this.config.baseUrl,
      hasApiKey: true,
    });
  }

  // ============================================================================
  // Header Construction
  // ============================================================================

  /**
   * Build request headers with authentication and attribution.
   * Returns a plain object for better compatibility with Node.js fetch.
   */
  buildHeaders(additionalHeaders?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.config.apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': this.config.httpReferer,
    };

    // Add any additional headers
    if (additionalHeaders) {
      Object.assign(headers, additionalHeaders);
    }

    // Debug log to verify Authorization header is present
    this.config.logger.debug('Request headers built', {
      hasAuthorization: Boolean(headers['Authorization']),
      authHeaderLength: headers['Authorization']?.length ?? 0,
    });

    return headers;
  }

  // ============================================================================
  // Generic Request Method
  // ============================================================================

  /**
   * Make a request to the OpenRouter API with error handling, caching, and throttling.
   */
  async request<T>(endpoint: string, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    const { method = 'GET', body, headers: additionalHeaders, timeout, skipThrottle = false } = options;

    const url = `${this.config.baseUrl}${endpoint}`;
    const requestTimeout = timeout ?? this.config.timeout;

    // Apply throttling unless skipped
    let throttleStatus: ThrottleStatus = { isThrottled: false };
    if (!skipThrottle) {
      throttleStatus = await this.rateLimitManager.waitForThrottle();
    }

    // Build headers
    const headers = this.buildHeaders(additionalHeaders);

    // Detailed debug logging for troubleshooting
    this.config.logger.debug('Making API request', {
      method,
      endpoint,
      url,
      throttled: throttleStatus.isThrottled,
      hasAuthHeader: Boolean(headers['Authorization']),
      authHeaderPrefix: headers['Authorization']?.substring(0, 20) + '...',
      allHeaderKeys: Object.keys(headers),
    });

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), requestTimeout);

    try {
      const fetchOptions: RequestInit = {
        method,
        headers,
        signal: controller.signal,
      };

      if (body) {
        fetchOptions.body = JSON.stringify(body);
      }

      const response = await fetch(url, fetchOptions);

      // Parse rate limit headers
      const rateLimits = this.rateLimitManager.parseHeaders(response.headers);

      // Handle non-OK responses
      if (!response.ok) {
        let errorBody: OpenRouterErrorResponse;
        try {
          errorBody = (await response.json()) as OpenRouterErrorResponse;
        } catch {
          errorBody = { message: `HTTP ${response.status}: ${response.statusText}` };
        }

        throw parseOpenRouterError(errorBody, response.status, response.headers);
      }

      // Parse successful response
      const data = (await response.json()) as T;

      this.config.logger.debug('API request successful', {
        endpoint,
        statusCode: response.status,
      });

      return {
        data,
        rateLimits,
        throttleStatus,
        cached: false,
      };
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }

      // Handle abort/timeout
      if (error instanceof Error && error.name === 'AbortError') {
        throw new ApiError({
          code: ErrorCode.API_TIMEOUT,
          message: `Request timed out after ${requestTimeout}ms`,
        });
      }

      // Handle network errors
      throw createNetworkError(error);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ============================================================================
  // API Endpoints
  // ============================================================================

  /**
   * Fetch all available models from OpenRouter.
   * Uses caching to reduce API calls.
   */
  async listModels(skipCache = false): Promise<ApiResponse<OpenRouterModel[]>> {
    const cacheKey = CacheManager.KEYS.MODEL_LIST;

    // Check cache first
    if (!skipCache) {
      const cached = this.cacheManager.get<OpenRouterModel[]>(cacheKey);
      if (cached) {
        return {
          data: cached,
          rateLimits: this.rateLimitManager.getCurrentLimits(),
          throttleStatus: { isThrottled: false },
          cached: true,
        };
      }
    }

    // Fetch from API
    const response = await this.request<ModelsListResponse>('/models');

    // Cache the result
    this.cacheManager.set(cacheKey, response.data.data, CacheManager.TTL.MODEL_LIST);

    return {
      data: response.data.data,
      rateLimits: response.rateLimits,
      throttleStatus: response.throttleStatus,
      cached: false,
    };
  }

  /**
   * Create a chat completion.
   */
  async createChatCompletion(
    request: ChatCompletionRequest
  ): Promise<ApiResponse<ChatCompletionResponse>> {
    return this.request<ChatCompletionResponse>('/chat/completions', {
      method: 'POST',
      body: request,
    });
  }

  /**
   * Create an image generation request.
   * Uses the /chat/completions endpoint with modalities parameter.
   */
  async createImageGeneration(
    request: ImageGenerationRequest
  ): Promise<ApiResponse<ImageGenerationResponse>> {
    return this.request<ImageGenerationResponse>('/chat/completions', {
      method: 'POST',
      body: {
        ...request,
        stream: false, // Image generation doesn't support streaming in the same way
      },
      timeout: 120000, // 2 minutes for image generation
    });
  }

  /**
   * Create a streaming chat completion.
   * Returns a ReadableStream for SSE chunks.
   */
  async createStreamingChatCompletion(
    request: ChatCompletionRequest
  ): Promise<{
    stream: ReadableStream<Uint8Array>;
    rateLimits: RateLimitInfo | null;
    throttleStatus: ThrottleStatus;
  }> {
    const url = `${this.config.baseUrl}/chat/completions`;

    // Apply throttling
    const throttleStatus = await this.rateLimitManager.waitForThrottle();

    const headers = this.buildHeaders();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({ ...request, stream: true }),
        signal: controller.signal,
      });

      // Parse rate limit headers
      const rateLimits = this.rateLimitManager.parseHeaders(response.headers);

      if (!response.ok) {
        let errorBody: OpenRouterErrorResponse;
        try {
          errorBody = (await response.json()) as OpenRouterErrorResponse;
        } catch {
          errorBody = { message: `HTTP ${response.status}: ${response.statusText}` };
        }
        throw parseOpenRouterError(errorBody, response.status, response.headers);
      }

      if (!response.body) {
        throw new ApiError({
          code: ErrorCode.API_RESPONSE_INVALID,
          message: 'No response body for streaming request',
        });
      }

      // Clear timeout since we got a response
      clearTimeout(timeoutId);

      return {
        stream: response.body,
        rateLimits,
        throttleStatus,
      };
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof ApiError) {
        throw error;
      }

      if (error instanceof Error && error.name === 'AbortError') {
        throw new ApiError({
          code: ErrorCode.API_TIMEOUT,
          message: `Streaming request timed out after ${this.config.timeout}ms`,
        });
      }

      throw createNetworkError(error);
    }
  }

  /**
   * Get generation stats for a specific completion.
   */
  async getGeneration(generationId: string): Promise<ApiResponse<GenerationStatsResponse>> {
    return this.request<GenerationStatsResponse>(`/generation?id=${encodeURIComponent(generationId)}`);
  }

  /**
   * Get all available endpoints/providers for a specific model.
   * @param modelSlug Model identifier in "author/model-name" format
   */
  async getModelEndpoints(modelSlug: string): Promise<ApiResponse<ModelEndpointsResponse>> {
    return this.request<ModelEndpointsResponse>(`/models/${modelSlug}/endpoints`);
  }

  // ============================================================================
  // Cache Management
  // ============================================================================

  /**
   * Get account credits information.
   */
  async getCredits(): Promise<ApiResponse<CreditsResponse>> {
    return this.request<CreditsResponse>('/key');
  }

  /**
   * Invalidate the model list cache.
   */
  invalidateModelCache(): void {
    this.cacheManager.invalidate(CacheManager.KEYS.MODEL_LIST);
    this.config.logger.debug('Model cache invalidated');
  }

  /**
   * Clear all caches.
   */
  clearCache(): void {
    this.cacheManager.clear();
    this.config.logger.debug('All caches cleared');
  }

  /**
   * Get cache statistics.
   */
  getCacheStats() {
    return this.cacheManager.getStats();
  }

  // ============================================================================
  // Rate Limit Management
  // ============================================================================

  /**
   * Get current rate limit information.
   */
  getRateLimits(): RateLimitInfo | null {
    return this.rateLimitManager.getCurrentLimits();
  }

  /**
   * Check current throttle status.
   */
  getThrottleStatus(): ThrottleStatus {
    return this.rateLimitManager.checkThrottle();
  }

  /**
   * Reset throttle state.
   */
  resetThrottle(): void {
    this.rateLimitManager.reset();
    this.config.logger.debug('Throttle state reset');
  }
}

export default OpenRouterClient;
