/**
 * OpenRouter MCP Server - Main server class following Perplexity MCP pattern.
 * Implements the Model Context Protocol for OpenRouter API access.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { Logger, logger as defaultLogger } from '../utils/logger.js';

// ============================================================================
// Types
// ============================================================================

export interface OpenRouterServerConfig {
  apiKey: string;
  logger?: Logger;
}

export interface TextContent {
  type: 'text';
  text: string;
}

export interface ToolRegistration {
  name: string;
  description: string;
  inputSchema: z.ZodType;
  handler: (args: unknown) => Promise<ToolResponse>;
}

export interface ToolResponse {
  content: TextContent[];
  structuredContent?: unknown;
  isError?: boolean;
}

export interface JsonSchema {
  type: 'object';
  properties?: Record<string, object>;
  required?: string[];
  [key: string]: unknown;
}

// ============================================================================
// Server Implementation
// ============================================================================

export class OpenRouterServer {
  private readonly server: Server;
  private readonly config: OpenRouterServerConfig;
  private readonly logger: Logger;
  private readonly tools: Map<string, ToolRegistration> = new Map();
  private transport: StdioServerTransport | null = null;
  private isRunning = false;

  constructor(config: OpenRouterServerConfig) {
    this.config = config;
    this.logger = config.logger ?? defaultLogger.child('server');

    // Initialize MCP Server
    this.server = new Server(
      {
        name: 'openrouter-mcp-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    // Set up request handlers
    this.setupRequestHandlers();

    this.logger.debug('OpenRouterServer initialized', {
      hasApiKey: Boolean(config.apiKey),
    });
  }

  /**
   * Set up MCP protocol request handlers
   */
  private setupRequestHandlers(): void {
    // Handle list tools request
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      this.logger.debug('Handling list tools request', {
        toolCount: this.tools.size,
      });

      const tools = Array.from(this.tools.values()).map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: this.zodToJsonSchema(tool.inputSchema),
      }));

      return { tools };
    });

    // Handle tool call request
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      this.logger.debug('Handling tool call request', { toolName: name });

      const tool = this.tools.get(name);
      if (!tool) {
        this.logger.error('Tool not found', { toolName: name });
        return {
          content: [
            {
              type: 'text' as const,
              text: `Error: Tool '${name}' not found`,
            },
          ],
          isError: true,
        };
      }

      try {
        // Validate input with Zod schema
        const parseResult = tool.inputSchema.safeParse(args);
        if (!parseResult.success) {
          const errorMessage = parseResult.error.issues
            .map((issue) => `${issue.path.join('.')}: ${issue.message}`)
            .join('; ');

          this.logger.warn('Tool input validation failed', {
            toolName: name,
            error: errorMessage,
          });

          return {
            content: [
              {
                type: 'text' as const,
                text: `Validation error: ${errorMessage}`,
              },
            ],
            isError: true,
          };
        }

        // Execute tool handler
        const response = await tool.handler(parseResult.data);

        this.logger.debug('Tool executed successfully', { toolName: name });

        return {
          content: response.content,
          isError: response.isError,
        };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        this.logger.error('Tool execution failed', {
          toolName: name,
          error: errorMessage,
        });

        return {
          content: [
            {
              type: 'text' as const,
              text: `Error executing tool: ${errorMessage}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  /**
   * Convert Zod schema to JSON Schema for MCP protocol
   */
  private zodToJsonSchema(schema: z.ZodType): JsonSchema {
    // Basic conversion - for complex schemas, consider using zod-to-json-schema library
    if (schema instanceof z.ZodEffects) {
      return this.zodToJsonSchema(schema.innerType());
    }
    if (schema instanceof z.ZodObject) {
      const shape = schema.shape as Record<string, z.ZodType>;
      const properties: Record<string, object> = {};
      const required: string[] = [];

      for (const [key, value] of Object.entries(shape)) {
        properties[key] = this.zodTypeToJsonSchema(value);
        if (!value.isOptional()) {
          required.push(key);
        }
      }

      return {
        type: 'object' as const,
        properties,
        required: required.length > 0 ? required : undefined,
      };
    }

    return { type: 'object' as const };
  }

  /**
   * Convert individual Zod type to JSON Schema type
   */
  private zodTypeToJsonSchema(zodType: z.ZodType): Record<string, unknown> {
    if (zodType instanceof z.ZodString) {
      return { type: 'string' };
    }
    if (zodType instanceof z.ZodNumber) {
      return { type: 'number' };
    }
    if (zodType instanceof z.ZodBoolean) {
      return { type: 'boolean' };
    }
    if (zodType instanceof z.ZodArray) {
      return {
        type: 'array',
        items: this.zodTypeToJsonSchema(zodType.element),
      };
    }
    if (zodType instanceof z.ZodOptional) {
      return this.zodTypeToJsonSchema(zodType.unwrap());
    }
    if (zodType instanceof z.ZodEnum) {
      return {
        type: 'string',
        enum: zodType.options,
      };
    }
    if (zodType instanceof z.ZodObject) {
      return this.zodToJsonSchema(zodType);
    }
    if (zodType instanceof z.ZodLiteral) {
      return { type: typeof zodType.value, const: zodType.value };
    }
    if (zodType instanceof z.ZodUnion) {
      return {
        anyOf: (zodType.options as z.ZodType[]).map((opt) =>
          opt instanceof z.ZodObject
            ? this.zodToJsonSchema(opt)
            : this.zodTypeToJsonSchema(opt),
        ),
      };
    }
    if (zodType instanceof z.ZodEffects) {
      // z.custom() produces ZodEffects — treat as string to allow passthrough
      return { type: 'string' };
    }

    return {};
  }

  /**
   * Register a tool with the server
   */
  registerTool(registration: ToolRegistration): void {
    if (this.tools.has(registration.name)) {
      this.logger.warn('Overwriting existing tool registration', {
        toolName: registration.name,
      });
    }

    this.tools.set(registration.name, registration);
    this.logger.info('Tool registered', { toolName: registration.name });
  }

  /**
   * Get the API key
   */
  getApiKey(): string {
    return this.config.apiKey;
  }

  /**
   * Check if server is running
   */
  isServerRunning(): boolean {
    return this.isRunning;
  }

  /**
   * Get registered tool names
   */
  getRegisteredTools(): string[] {
    return Array.from(this.tools.keys());
  }

  /**
   * Start the server with stdio transport
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      this.logger.warn('Server is already running');
      return;
    }

    this.logger.info('Starting OpenRouter MCP server...');

    try {
      // Create stdio transport
      this.transport = new StdioServerTransport();

      // Connect server to transport
      await this.server.connect(this.transport);

      this.isRunning = true;
      this.logger.info('OpenRouter MCP server started successfully', {
        transport: 'stdio',
        toolCount: this.tools.size,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.logger.error('Failed to start server', { error: errorMessage });
      throw error;
    }
  }

  /**
   * Stop the server
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      this.logger.warn('Server is not running');
      return;
    }

    this.logger.info('Stopping OpenRouter MCP server...');

    try {
      await this.server.close();
      this.transport = null;
      this.isRunning = false;
      this.logger.info('OpenRouter MCP server stopped');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.logger.error('Error stopping server', { error: errorMessage });
      throw error;
    }
  }
}

export default OpenRouterServer;
